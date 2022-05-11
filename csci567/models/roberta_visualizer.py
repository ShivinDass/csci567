import numpy as np
import os
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
import torch
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
tqdm.pandas()


class TSNEVisualizer:
    def __init__(self, embeddings, labels=None, label_tags=None, n_jobs=1):
        """ Visualizes embeddings in a 2D TSNE embedding space.
        :arg embeddings: [n_elem, embedding_dim] numpy array
        :arg labels: (optional) [n_elem,] numpy array of label indices
        :arg label_tags: (optional) dict of label indices and corresponding tag for plot legend
        :arg n_jobs: number of workers used for TSNE embedding computation, default: 1
        """
        self.embeddings = embeddings
        self.labels = labels
        self.label_tags = label_tags
        self.tsne = TSNE(n_components=2, perplexity=50, n_iter=2000,
                         init='pca', learning_rate='auto', n_jobs=n_jobs, verbose=False)
        self.tsne_data = self.tsne.fit_transform(self.embeddings)

    def visualize(self):
        fig = plt.figure(figsize=(8, 5), dpi=50)
        labels = self.labels if self.labels is not None else np.zeros_like(
            self.embeddings[:, 0])
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            if self.labels is None:
                idxs = np.arange(self.embeddings.shape[0])
            else:
                idxs = np.where(self.labels == label)
            plt.scatter(self.tsne_data[idxs, 0],
                        self.tsne_data[idxs, 1],
                        label=self.label_tags[label] if self.label_tags is not None else label,
                        s=10)
        if self.labels is not None:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                       fancybox=True, shadow=True, ncol=6, fontsize=9)
            plt.tight_layout()
        plt.show()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base").to(device)


def wrapped_tokenizer(x):
    if isinstance(x, str):
        return tokenizer(
            x, add_special_tokens=True, truncation=True,
            padding="max_length", return_attention_mask=True, return_tensors="pt")
    print("returning None")
    return None


def wrapped_bert(x):
    try:
        for k in x:
            x[k] = x[k].to(device)
        with torch.no_grad():
            return model(**x).last_hidden_state[:, 0, :][0].cpu().numpy()
    except ValueError:
        return None


articles_df = pd.read_csv(os.path.join(os.environ["DATA_DIR"], "articles.csv"))
articles_df.fillna('', inplace=True)
articles_desc = articles_df['product_group_name'] + " " + \
    articles_df['product_type_name'] + " " + \
    articles_df['graphical_appearance_name'] + " " + \
    articles_df['colour_group_name'] + " " + \
    articles_df['index_group_name'] + " " + \
    articles_df['graphical_appearance_name'] + " " + \
    articles_df['detail_desc']

tokenized_desc = articles_desc.progress_apply(wrapped_tokenizer)
embedded_desc = tokenized_desc.progress_apply(wrapped_bert)

embedding_df = pd.DataFrame()
embedding_df['article_id'] = articles_df['article_id']
embedding_df['desc_embedding'] = embedded_desc
embedding_df.to_pickle("article_roberta_embedding.pkl")

article_group = articles_df['product_group_name'].to_numpy()
embedded_desc_np = embedded_desc.to_numpy()
embedded_desc_np = np.stack(embedded_desc_np)

print("Running TSNE")
vis = TSNEVisualizer(embedded_desc_np[::10], labels=article_group[::10])
vis.visualize()
