# import networkx as nx
# import pandas as pd
# import numpy as np
# import torch
# from sklearn.cluster import SpectralClustering
#
# # Load the adjacency matrix again
# adj_matrix = pd.read_csv('../data/A_adj(13291329).csv', header=None).to_numpy()
#
# # Assuming the adjacency matrix is symmetric and represents an undirected graph
# # We will create an edge list from the adjacency matrix
# rows, cols = np.where(adj_matrix == 1)
# edges = list(zip(rows.tolist(), cols.tolist()))
#
# # Remove duplicate edges (since the matrix is symmetric)
# edges = list(set([tuple(sorted(edge)) for edge in edges if edge[0] != edge[1]]))
#
# # Create a graph from the edge list
# G = nx.Graph()
# G.add_edges_from(edges)
#
# # Use spectral clustering to infer hyperedges (200 hyperedges as per the requirement)
# sc = SpectralClustering(n_clusters=200, affinity='precomputed', n_init=100, assign_labels='discretize')
# sc.fit(adj_matrix)
#
# # The labels indicate the hyperedge each node belongs to
# hyperedge_labels = sc.labels_
#
# # Generate hyperedge index
# hyperedge_indices = []
# node_indices = []
#
# for node_id, hyperedge_id in enumerate(hyperedge_labels):
#     hyperedge_indices.append(hyperedge_id)
#     node_indices.append(node_id)
#
# # Convert to numpy arrays and format as hyperedge index
# hyperedge_indices_np = np.array(hyperedge_indices)
# node_indices_np = np.array(node_indices)
# hyperedge_index_np = np.vstack((hyperedge_indices_np, node_indices_np))
# hyperedge_index_tensor = torch.tensor(hyperedge_index_np).clone().detach()
#
# # Display the shape of the hyperedge index
# print(hyperedge_index_np.shape)
