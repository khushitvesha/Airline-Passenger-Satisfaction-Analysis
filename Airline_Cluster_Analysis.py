#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 17:47:18 2024

@author: leticiatca
"""

import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv('cleaned_airline_satisfaction_data.csv')

#Transform categorical columns into binary
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

df_dummies = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

#Create Clusters
features = ['Age', 'Flight Distance', 'Gender_Male', 'Customer Type_disloyal Customer',
'Type of Travel_Personal Travel', 'Class_Eco', 'Class_Eco Plus',
'satisfaction_satisfied']

num_clusters = 3  
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Fit the K-means model to the data
kmeans.fit(df_dummies[features])

# Add a new column 'Cluster' to the DataFrame indicating the cluster each customer belongs to
df_dummies['Cluster'] = kmeans.labels_

# Optionally, you can inspect the centroids of the clusters
centroids = kmeans.cluster_centers_
print("Centroids of the clusters:")
print(centroids)

clusters=df_dummies.groupby('Cluster').mean()

transposed_df = clusters.transpose()

transposed_df.to_csv('clusters2.csv', index=False)
