import os
import pickle
import numpy as np

from sklearn.decomposition import PCA


if __name__ == '__main__':

    dataname = 'Electronics'
    dim = 128

    with open('./data/processed/' + dataname + '_embedding.pkl', 'rb') as f:
        llm_embedding = pickle.load(f)

    pca = PCA(n_components=dim)
    pca_embedding = pca.fit_transform(llm_embedding)

    with open('./data/processed/' + dataname + '_pca.pkl', 'wb') as f:
        pickle.dump(pca_embedding, f, pickle.HIGHEST_PROTOCOL)




