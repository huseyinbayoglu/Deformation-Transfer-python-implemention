"""
Reading, writing obj files and some util functions like vertex_to_triangles or build_face_adjacency
"""

import numpy as np 
from collections import defaultdict


def load_obj(file_path):
    "Get Vertices and triangles from a .obj file"
    vertices = []
    faces = []

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):  
                parts = line.strip().split()
                v = list(map(float, parts[1:4]))
                vertices.append(v)

            elif line.startswith('f '):  # face bunlarda "f "ile başlar böylece mesh oluşur
                parts = line.strip().split()
                
                # f 1 2 3  veya  f 1/1/1 2/2/2 3/3/3
                face = []
                for p in parts[1:]:
                    idx = p.split('/')[0]  # sadece vertex index al
                    face.append(int(idx) - 1)  # obj 1-based → 0-based

                faces.append(face)

    return (np.array(vertices), np.array(faces))



def write_obj(output_file, vertices, faces):
    with open(output_file, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write("f " + " ".join(str(idx + 1) for idx in face) + "\n")


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

def vertex_to_triangles(faces):
    """
    Her vertex'in hangi üçgenlerde kullanıldığını döndürür — p(v).
    faces: (F, 3)
    return: dict {vertex_index: [face_index, ...]}
    """
    vtx_to_tri = defaultdict(list)
    for face_idx, face in enumerate(faces):
        for v_idx in face:
            vtx_to_tri[int(v_idx)].append(face_idx)
    return vtx_to_tri


