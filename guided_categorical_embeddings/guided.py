from typing import List, Optional, Union, Tuple, Dict

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.neural_network import MLPClassifier


class LabelGuidedEmbeddingsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 hidden_layer_sizes: Tuple[int, ...] = (10, 5),
                 hidden_layer_index: int = 0,
                 embedding_size: Optional[int] = None,
                 max_iter: int = 1000,
                 random_state: int = 42,
                 vectorizer: DictVectorizer = None):
        """
        Initializes the GuidedEmbeddingsTransformer with the given parameters.

        Parameters:
        - hidden_layer_sizes: tuple, the sizes of the hidden layers in the MLP.
        - hidden_layer_index: int, the index of the hidden layer to use for embeddings.
        - embedding_size: int or None, the size of the output embedding. If None, PCA will not be applied.
        - max_iter: int, the maximum number of iterations for the MLP training.
        - random_state: int, seed for random number generator.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layer_index = hidden_layer_index
        self.embedding_size = embedding_size
        self.max_iter = max_iter
        self.random_state = random_state
        self.vectorizer = vectorizer or DictVectorizer(sparse=False)
        self.mlp = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes,
                                 max_iter=self.max_iter,
                                 random_state=self.random_state)
        self.pca: Optional[PCA] = None  # Initialize PCA later based on data

    def predict(self, X: Union[np.ndarray, List[Dict[str, str]]]) -> List[str]:
        if isinstance(X[0], dict):
            X = self.vectorizer.transform(X)
        return self.mlp.predict(X)

    def fit(self, X: Union[np.ndarray, List[Dict[str, str]]], y: List[str]) -> 'LabelGuidedEmbeddingsTransformer':
        """
        Fits the MLP model and learns the embedding space. Optionally applies PCA if embedding_size is specified.

        Parameters:
        - X: array-like, shape (n_samples, n_features), feature matrix.
        - y: array-like, guide domain relevant dataset, target labels.

        Returns:
        - self: returns an instance of self.
        """
        if isinstance(X[0], dict):
            X = self.vectorizer.fit_transform(X)
        self.mlp.fit(X, y)
        hidden_activations = self._get_hidden_activations(X)
        self.embedding_size = self.embedding_size or 3 * len(set(y))
        if self.embedding_size:
            # Ensure embedding size does not exceed the number of features in hidden activations
            n_features = hidden_activations.shape[1]
            n_components = min(self.embedding_size, n_features)
            self.pca = PCA(n_components=n_components)
            self.pca.fit(hidden_activations)
        return self

    def transform(self, X: Union[np.ndarray, List[Dict[str, str]]]) -> np.ndarray:
        """
        Transforms the input data into embeddings.

        Parameters:
        - X: array-like, shape (n_samples, n_features), feature matrix.

        Returns:
        - Transformed data as embeddings.
        """
        if isinstance(X[0], dict):
            X = self.vectorizer.transform(X)
        hidden_activations = self._get_hidden_activations(X)
        if self.embedding_size:
            hidden_activations = self.pca.transform(hidden_activations)
        return hidden_activations

    def _get_hidden_activations(self, X: np.ndarray) -> np.ndarray:
        """
        Extracts hidden layer activations from the MLP model.

        Parameters:
        - X: array-like, shape (n_samples, n_features), feature matrix.

        Returns:
        - Hidden layer activations as a numpy array.
        """
        coefs = self.mlp.coefs_
        intercepts = self.mlp.intercepts_

        activation = X
        for i in range(self.hidden_layer_index + 1):
            activation = np.dot(activation, coefs[i]) + intercepts[i]
            activation = np.maximum(activation, 0)  # ReLU activation
        return activation

    def save(self, filepath: str) -> None:
        """
        Saves the transformer to a file.

        Parameters:
        - filepath: str, path to the file where the transformer will be saved.
        """
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'LabelGuidedEmbeddingsTransformer':
        """
        Loads the transformer from a file.

        Parameters:
        - filepath: str, path to the file from which the transformer will be loaded.

        Returns:
        - An instance of GuidedEmbeddingsTransformer.
        """
        return joblib.load(filepath)


class MultiLabelGuidedEmbeddingsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 n_embedders: int = 2,
                 hidden_layer_sizes_list: Optional[List[Tuple[int, ...]]] = None,
                 hidden_layer_index_list: Optional[List[int]] = None,
                 embedding_size_list: Optional[List[Optional[int]]] = None,
                 max_iter: int = 1000,
                 random_state: int = 42):
        """
        Initializes the MultiDatasetGuidedEmbeddingsTransformer with the given parameters.

        Parameters:
        - n_embedders: int, the number of embedding layers.
        - hidden_layer_sizes_list: list of tuples, sizes of hidden layers for each embedder.
        - hidden_layer_index_list: list of ints, indices of hidden layers to use for embeddings for each embedder.
        - embedding_size_list: list of ints or None, sizes of the output embeddings for each embedder.
        - max_iter: int, the maximum number of iterations for the MLP training.
        - random_state: int, seed for random number generator.
        """
        self.n_embedders = n_embedders
        self.hidden_layer_sizes_list = hidden_layer_sizes_list or [(128, 64)] * n_embedders
        self.hidden_layer_index_list = hidden_layer_index_list or [0] * n_embedders
        self.embedding_size_list = embedding_size_list or [None] * n_embedders
        self.max_iter = max_iter
        self.random_state = random_state
        self.vectorizer = DictVectorizer(sparse=False)

        # Ensure lists are of correct length
        assert len(self.hidden_layer_sizes_list) == n_embedders
        assert len(self.hidden_layer_index_list) == n_embedders
        assert len(self.embedding_size_list) == n_embedders

        # Initialize the list of embedders
        self.embedders = [
            LabelGuidedEmbeddingsTransformer(
                hidden_layer_sizes=self.hidden_layer_sizes_list[i],
                hidden_layer_index=self.hidden_layer_index_list[i],
                embedding_size=self.embedding_size_list[i],
                max_iter=self.max_iter,
                random_state=self.random_state,
                vectorizer=self.vectorizer
            ) for i in range(n_embedders)
        ]

    def predict(self, X: Union[np.ndarray, List[Dict[str, str]]], index:int) -> List[str]:
        return self.embedders[index].predict(X)

    def _get_internal_embeddings(self, X: Union[np.ndarray, List[Dict[str, str]]], index: int) -> np.ndarray:
        """
        Gets the transformed embeddings from a specific embedder.

        Parameters:
        - X: array-like, shape (n_samples, n_features), feature matrix.
        - index: int, index of the embedder.

        Returns:
        - Embeddings from the specified embedder.
        """
        return self.embedders[index].transform(X)

    def fit(self, X: Union[np.ndarray, List[Dict[str, str]]],
            y: List[np.ndarray]) -> 'MultiLabelGuidedEmbeddingsTransformer':
        """
        Fits each MLP embedder and learns the embedding space for multiple datasets.

        Parameters:
        - X: array-like, shape (n_samples, n_features), feature matrix.
        - y: list of array-like, guide datasets, target values for each dataset.

        Returns:
        - self: returns an instance of self.
        """
        assert len(y) == self.n_embedders
        transformed_X = self.vectorizer.fit_transform(X)

        for i in range(self.n_embedders):
            self.embedders[i].fit(transformed_X, y[i])
            embeddings = self._get_internal_embeddings(transformed_X, index=i)
            # Concatenate embeddings with original features
            transformed_X = np.hstack([transformed_X, embeddings])

        return self

    def transform(self, X: np.ndarray, layer: Optional[int] = None) -> np.ndarray:
        """
        Transforms the input data into embeddings using all configured embedders.

        Parameters:
        - X: array-like, shape (n_samples, n_features), feature matrix.
        - layer: int, from which guide dataset index to get the embeddings

        Returns:
        - Transformed data as embeddings.
        """
        transformed_X = self.vectorizer.transform(X)

        for i in range(self.n_embedders):
            embeddings = self._get_internal_embeddings(transformed_X, index=i)
            if layer is not None and layer == i:
                return embeddings
            if i < self.n_embedders - 1:
                transformed_X = np.hstack([transformed_X, embeddings])

        return transformed_X

    def save(self, filepath: str) -> None:
        """
        Saves the transformer to a file.

        Parameters:
        - filepath: str, path to the file where the transformer will be saved.
        """
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'MultiLabelGuidedEmbeddingsTransformer':
        """
        Loads the transformer from a file.

        Parameters:
        - filepath: str, path to the file from which the transformer will be loaded.

        Returns:
        - An instance of MultiDatasetGuidedEmbeddingsTransformer.
        """
        return joblib.load(filepath)
