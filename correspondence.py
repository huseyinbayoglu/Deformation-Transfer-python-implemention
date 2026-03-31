"""
Computes the correspondence between vertices of two models.
It "inflates" the source mesh until it fits the target mesh (by minimizing a cost function).
"""

import tqdm 
import numpy as np 
import time

import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
from scipy.spatial import cKDTree

from collections import defaultdict

from phase1 import load_obj

def build_edge_to_faces(faces:np.array):
    edge_to_faces = defaultdict(list)

    for i,f in enumerate(faces):
        v0,v1,v2 = f # vertices indexes

        edges = [
            tuple(sorted((v0,v1))),
            tuple(sorted((v0,v2))),
            tuple(sorted((v1,v2))),
        ]

        for e in edges:
            edge_to_faces[e].append(i)

    return edge_to_faces


def build_face_adjacency(edge_to_faces, num_faces):
     adjacency = [[] for _ in range(num_faces)]

     for edge, face_list in edge_to_faces.items():
         if len(face_list) < 2: # boundary edge
             continue 
         
         for f in face_list:
             for neighbor in face_list:
                 if f != neighbor:
                     adjacency[f].append(neighbor)

     return adjacency

def remove_duplicates(adjacency):
    return [list(set(neigh)) for neigh in adjacency]

def compute_adjacent_by_edges(mesh):
    V, F = mesh

    edge_to_faces = build_edge_to_faces(F)
    adjacency = build_face_adjacency(edge_to_faces, len(F))
    adjacency = remove_duplicates(adjacency)

    return adjacency






# TEST
if __name__ == "__main__":
    source_obj_path = "source_obj/cat-poses/cat-01.obj"
    mesh = load_obj(source_obj_path)
    t1 = time.time()
    adjaceny = compute_adjacent_by_edges(mesh)
    print(adjaceny[:20])
    print(f"Calculation takes {(time.time()-t1):.2f} seconds")