import modal

from dataclasses import dataclass
from itertools import product
from helpers.data import ProductDataset, prepare_eval_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

import pandas as pd

app = modal.App('finetune-embedding-model')

gpu_config = modal.gpu.A100()

### Constants ###
MODEL = "sentence-transformers/all-mpnet-base-v2"
SCHEDULER = "warmuplinear"
DATASET_SIZE = [4776]
WARMUP_STEPS = [100]
DENSE_OUT_FEATURES = [512] #64, 128, 256, 
BATCH_SIZE = [32]
MODEL_VOLUME = modal.Volume.from_name("models", create_if_missing=True)
MAX_EPOCHS = 8

DATASET_NAME = "lakritidis/product-matching"
DATASET_CONFIG = "default"
METRICS = {
    "accuracy": accuracy_score,  # This is the number of correct predictions by the model ( TP + TN )/ (# of samples)
    "precision": precision_score,  # This measures the number of positive class predicitons (TP) / (TP + FP)
    "recall": recall_score,  # This measure the number of negative class predictions (TP) / ( TP + FN )
    "AUC": roc_auc_score,
}


### Utils ###
def download_model():
    from sentence_transformers import SentenceTransformer
    SentenceTransformer(MODEL)

def preprocess_data(df) -> pd.DataFrame:
    df = df[['Product Title', 'Cluster Label']]
    value_counts = df['Cluster Label'].value_counts()
    filtered_labels = value_counts[value_counts >= 10].index
    df = df[df['Cluster Label'].isin(filtered_labels)]
    df = df.rename(columns={
        'Cluster Label': 'label',
        'Product Title': 'title'
    })
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df = train_df.head(4768)
    return train_df, test_df

def generate_configs():
    for (
        sample_size,
        freeze_embedding_model,
        dense_out_features,
    ) in product(DATASET_SIZE, [True, False], DENSE_OUT_FEATURES):
        yield ModelConfig(
            model_name=MODEL,
            dataset_size=sample_size,
            freeze_embedding_model=freeze_embedding_model,
            dense_out_features=dense_out_features,
            learning_rate=1e-4,
            batch_size=32,
            warmup_steps=500,
            scheduler="warmuplinear",
            num_epochs=8,
        )

### Modal ###
image = (
    modal.Image.debian_slim()
    .pip_install("sentence-transformers", "torch", "datasets", "pandas", "transformers[torch]")
    .run_function(download_model)
)

@dataclass
class ModelConfig:
    model_name: str
    dataset_size: int
    dense_out_features: int
    learning_rate: float
    scheduler: str
    warmup_steps: int
    freeze_embedding_model: bool
    batch_size: int
    num_epochs: int

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu=gpu_config,
    volumes={"/models": MODEL_VOLUME},
    concurrency_limit=50,
    allow_concurrent_inputs=True,
    timeout=86400,
)
def objective(
    config: ModelConfig,
):
    from sentence_transformers import SentenceTransformer, losses, evaluation, models
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    import torch.nn as nn

    model_name = config.model_name
    dense_out_features = config.dense_out_features
    learning_rate = config.learning_rate
    scheduler = config.scheduler
    warmup_steps = config.warmup_steps
    freeze_embedding_model = config.freeze_embedding_model
    batch_size = config.batch_size
    num_epochs = config.num_epochs

    print(f"Training model {model_name} {config}")

    # Load the model
    embedding_model = SentenceTransformer(model_name)

    # Model configuration
    dim_emb = embedding_model.get_sentence_embedding_dimension()

    # Freeze the embedding model
    if freeze_embedding_model:
        for param in embedding_model._first_module().auto_model.parameters():
            param.requires_grad = False

    # Define the model architecture with additional dense layer
    dense_model = models.Dense(
        in_features=dim_emb,
        out_features=dense_out_features,
        activation_function=nn.Tanh(),
    )
    pooling_model = models.Pooling(dim_emb)

    # Initialize the model
    model = SentenceTransformer(
        modules=[embedding_model, pooling_model, dense_model], device="cuda"
    )

    # Load the dataset
    dataset = load_dataset("lakritidis/product-matching")
    df = dataset['train'].to_pandas()

    train_df, test_df = preprocess_data(df)
    train_dataset = ProductDataset(train_df)
    queries, corpus, relevant_docs = prepare_eval_data(test_df)
    
    # Create dataloaders and evaluator
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    evaluator = evaluation.InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="Product-Category-Retrieval-Test",
    show_progress_bar=True,
    )

    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        warmup_steps=warmup_steps,
        scheduler=scheduler,
        optimizer_params={"lr": learning_rate},
        save_best_model=True,
        epochs=num_epochs,
        output_path="/models",
    )

    # Reload the best model
    model = SentenceTransformer("/models")
    # Safe model to HF for later use
    model.push_to_hub("embedding_finetune")
    results = evaluator(model)
    print(results[evaluator.primary_metric])
    return results

@app.local_entrypoint()
def main():

    results = []
    for experiment_result in objective.map(
        generate_configs(), order_outputs=True, return_exceptions=True
    ):
        if isinstance(experiment_result, Exception):
            print(f"Encountered Exception of {experiment_result}")
            continue
        results.append(experiment_result)
    print(f'training done:{results}')