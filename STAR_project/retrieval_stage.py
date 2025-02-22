#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:07:54 2025

@author: rcyuh
"""

"""
Đang có: 
    embeddings => 1 dict với key là itemID và value là embedding vector tương ứng (chứa trong ndarray)
    collab_matrix => 1 ndarray với chiều NxN (N là số items)
    
Cần làm:
    tính sematic_matrix: sử dụng embeddings => OK
    tính scores: sử dụng semantic_matrix, collab_matrix, 3 factors:
        B1: Với 1 user u => Phải thu được interaction_history của nó 
            => thứ tự kiểu gì? dùng timestamp hay ratings để sắp xếp? => timestamp
            Thu được interaction_history đồng nghĩa biết được unseen items
        B2: Với mỗi unseen item => tính theo công thức trong paper
        
    Output của stage này maybe sẽ là với mỗi user => top k unseen items (theo score)
    ??? Lưu trữ bằng gì: Dict (key - userID, val - top k)
    ??? Top k là list chứa các itemID ???

"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.spatial.distance import cosine, cdist
from scipy.sparse import csr_matrix
from tqdm import tqdm

class STARRetrieval:
    def __init__(self, 
                 semantic_weight: float = 0.5,    
                 temporal_decay: float = 0.7,     
                 history_length: int = 3):        
        self.semantic_weight = semantic_weight
        self.temporal_decay = temporal_decay
        self.history_length = history_length
        
        self.semantic_matrix = None
        self.collaborative_matrix = None
        self.item_to_idx = {}
        self.idx_to_item = {}
        
    def compute_semantic_relationships(self, 
                                       item_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute semantic similarity matrix from item embeddings"""
        print("\nComputing semantic relationships...")
        
        sorted_items = sorted(item_embeddings.keys())
        self.item_to_idx = {item: idx for idx, item in enumerate(sorted_items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        n_items = len(self.item_to_idx)
        
        # Convert embeddings to array and normalize
        embeddings_array = np.zeros((n_items, next(iter(item_embeddings.values())).shape[0]))
        for item_id, embedding in item_embeddings.items():
            embeddings_array[self.item_to_idx[item_id]] = embedding
            
        # L2 normalize embeddings
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        embeddings_array = embeddings_array / norms
        
        # Compute similarities using cosine distance (cosine dist # cosine similarity)
        semantic_matrix = 1 - cdist(embeddings_array, embeddings_array, metric='cosine')
        np.fill_diagonal(semantic_matrix, 0)
        
        # normalize: trong contex này, Đối lập (< 0) =  Không liên quan (0)
        self.semantic_matrix = np.maximum(0, semantic_matrix)
        
        return self.semantic_matrix