# {ROOT_DIR}/ironclad/modules/retrieval/search.py

# Delegates search to the index object (e.g., FaissBruteForce), which handles the FAISS logic.

import numpy as np
import torch # Needed for handling torch.Tensor query_embedding

from modules.retrieval.index.bruteforce import FaissBruteForce 

class FaissSearch:
    """
    Performs search operations on a given index object.
    It expects the 'index_obj' passed to it to have a 'search' method
    that handles the metric and returns distances, indices, and metadata.
    """
    def __init__(self, index_obj: FaissBruteForce, metric: str):
        """
        Initialize the search class with an index instance and distance metric.
        
        :param index_obj: An instance of an index class (e.g., FaissBruteForce, FaissHNSW, FaissLSH)
                          that has an 'add_embeddings' and 'search' method.
        :param metric: The distance metric ('euclidean', 'dot_product', 'cosine').
                       This metric will be passed to the index_obj's search method.
        """
        self.index_obj = index_obj
        self.metric = metric.lower() # Store the metric, as the index_obj's search method might use it.
        print(f"FaissSearch initialized with index of type {type(index_obj).__name__} and metric {self.metric}.")

    def search(self, query_vector: torch.Tensor, k: int = 5):
        """
        Perform a nearest neighbor search and retrieve the associated metadata.
        
        :param query_vector: The vector to query (torch.Tensor or numpy array).
        :param k: Number of nearest neighbors to return.
        :return: Tuple of (distances, indices, metadata) for the nearest neighbors.
                 Distances and indices are numpy arrays, metadata is a list of lists.
        """
        # The actual search logic, including metric handling and metadata retrieval,
        # is now delegated entirely to the index_obj's search method.
        # This keeps FaissSearch simple and compatible with various index implementations.
        return self.index_obj.search(query_vector, k)
