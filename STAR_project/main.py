#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:31:03 2025

@author: rcyuh
"""

import pandas as pd
import gzip
import json
from ast import Dict, Set
import os
from typing import List, Dict, Set
import numpy as np
from pathlib import Path
from collaborative_commonality import CollaborativeRelationshipProcessor
from item_embedding import ItemEmbeddingGenerator
from preprocessing import preProcessing_metadata, preProcessing_user_item_matrix
from retrieval_stage import STARRetrieval

def parse(path):
    g = open(path, 'r', encoding="utf-8")
    for l in g:
        yield json.loads(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
      df[i] = d
      i += 1
    return pd.DataFrame.from_dict(df, orient='index')

# CONST
path = "/home/rcyuh/Desktop/2. BAI/Quá trình học/Bước 3/data/"

meta_data_df = getDF(path + 'Beauty_Metadata.json.gz')
review_df = getDF(path + 'Beauty_Review.json.gz')
user_item_df = pd.read_csv(path + "Beauty_User-Item_Matrix.csv", names=["itemID", "userID", "rating", "timestamp"], header=None)

# Get item embeddings
generator = ItemEmbeddingGenerator()
item_embeddings, item_to_idx = generator.load_embeddings()

# # Compute semantic relationship
# retrieval = STARRetrieval()
# semantic_matrix = retrieval.compute_semantic_relationships(item_embeddings)

# # Tính collaborative matrix
# preprocess_matrix = preProcessing_user_item_matrix(user_item_df)
# user_history = preprocess_matrix.interaction_history
# rating_history = preprocess_matrix.rating_history 
# interactions = preprocess_matrix.convert_df_to_list(user_item_df)

# collab_processor = CollaborativeRelationshipProcessor()
# collab_processor.process_interactions(interactions = interactions, item_mapping = item_to_idx)

# Tính scores for unseen items





























