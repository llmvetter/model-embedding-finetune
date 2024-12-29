import random
from torch.utils.data import Dataset


def format_dataset(dataset):
    from sentence_transformers import InputExample
    return [
        InputExample(
            texts=row["questions"]["text"], label=1 if row["is_duplicate"] else 0
        )
        for row in dataset
    ]


class InputExample:
    def __init__(self, texts, label):
        self.texts = texts
        self.label = label

class ProductDataset(Dataset):
    def __init__(self, df, num_negatives=20):
        self.texts = df['title'].tolist()
        self.labels = df['label'].tolist()
        self.num_negatives = num_negatives
        self.all_labels = list(set(self.labels))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        anchor = self.texts[idx]
        positive = self.labels[idx]
        negatives = random.sample([label for label in self.all_labels if label != positive], 
                                  k=min(self.num_negatives, len(self.all_labels)-1))
        return InputExample(texts=[anchor, positive] + negatives, label=0)


def prepare_eval_data(df):
    queries = dict(zip(df.index.astype(str), df['title']))
    unique_categories = set(df['label'])
    corpus = dict(zip(map(str, range(len(unique_categories))), unique_categories))

    relevant_docs = {}
    for idx, row in df.iterrows():
        query_id = str(idx)
        relevant_category_id = str(list(corpus.values()).index(row['label']))
        relevant_docs[query_id] = set([relevant_category_id])
    return queries, corpus, relevant_docs
