import gradio as gr
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

#load model
embedder = SentenceTransformer(
    model_name_or_path="llmvetter/embedding_finetune",
)
with open('unique_labels.txt', 'r') as file:
    labels = [line.strip() for line in file]

label_embeddings = embedder.encode(labels, convert_to_tensor=True)

def classify(query:str, top_k:int=5):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    similarity_scores = embedder.similarity(query_embedding, label_embeddings)[0]
    scores, indices = torch.topk(similarity_scores, k=top_k)
    result_dict =  {labels[idx]: round(float(score), 3) for score, idx in zip(scores, indices)}
    df = pd.DataFrame(list(result_dict.items()), columns=['category', 'score'])
    return df

demo = gr.Interface(
    fn=classify,
    inputs=[gr.Textbox(lines=10, label="Base Product")],
    outputs=[gr.Dataframe(headers=["category", "score"], row_count=1)],
    title="Classifier",
    description="Product classification using fineturned mpnet",
    allow_flagging="never",
)


demo.launch()