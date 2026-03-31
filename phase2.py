import numpy as np
from phase1 import load_obj


def calculate_fourth_vertex(vertices, faces):
    """
    For every triangle, calculates 4th corner (Equation1)
    """
    v1 = vertices[faces[:, 0]]  # (F, 3)
    v2 = vertices[faces[:, 1]]  # (F, 3)
    v3 = vertices[faces[:, 2]]  # (F, 3)

    cross = np.cross(v2 - v1, v3 - v1)  # (F, 3)
    cross_norm = np.linalg.norm(cross, axis=1, keepdims=True)  # (F, 1)

    v4 = v1 + cross / np.sqrt(cross_norm)
    return v4


def get_V(vertices, faces):
    """
    V = [v2-v1, v3-v1, v4-v1]  (Equation 3).

    return: (F, 3, 3) — 3x3 matris for every triangle/face
    """
    v1 = vertices[faces[:, 0]]  # (F, 3)
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    v4 = calculate_fourth_vertex(vertices, faces)

    # V columns: (v2-v1), (v3-v1), (v4-v1)
    # V[i] shape (3,3)
    V = np.stack([v2 - v1, v3 - v1, v4 - v1], axis=2)  # (F, 3, 3)
    return V


def compute_source_deformations(ref_vertices, ref_faces, def_vertices):
    """
    Computes the affine transformations mapping the source mesh from the 
    reference to the deformed configuration.

    Q = Ṽ V⁻¹  (Equation 4)

    ref_vertices: (N, 3)  vertices of the reference mesh  
    ref_faces: (F, 3)  triangle indices 
    def_vertices: (N, 3)  vertices of the deformed mesh  

    return: (F, 3, 3)  a 3×3 transformation matrix S for each triangle
    """
    V = get_V(ref_vertices, ref_faces)          # (F, 3, 3)
    V_tilde = get_V(def_vertices, ref_faces)    # (F, 3, 3)

    V_inv = np.linalg.inv(V)                    # (F, 3, 3)
    Q = V_tilde @ V_inv                         # (F, 3, 3) — Q = Ṽ V⁻¹

    return Q

