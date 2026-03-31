import numpy as np
from phase1 import load_obj


def calculate_fourth_vertex(vertices, faces):
    """
    Her üçgen için normal yönünde sanal 4. vertex hesaplar (Equation 1).

    vertices: (N, 3) — tüm mesh vertex'leri
    faces: (F, 3) — üçgen indeksleri

    return: (F, 3) — her üçgen için 4. vertex koordinatları
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
    V = [v2-v1, v3-v1, v4-v1] matrisini her üçgen için hesaplar (Equation 3).

    return: (F, 3, 3) — her üçgen için 3x3 V matrisi
    """
    v1 = vertices[faces[:, 0]]  # (F, 3)
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    v4 = calculate_fourth_vertex(vertices, faces)

    # V sütunları: (v2-v1), (v3-v1), (v4-v1)
    # V[i] shape (3,3): her satır bir sütun vektörü (transpose ile düzeltiriz)
    V = np.stack([v2 - v1, v3 - v1, v4 - v1], axis=2)  # (F, 3, 3)
    return V


def compute_source_deformations(ref_vertices, ref_faces, def_vertices):
    """
    Source mesh'in reference → deformed affine dönüşümlerini hesaplar.
    Q = Ṽ V⁻¹  (Equation 4)

    ref_vertices: (N, 3) — reference mesh vertex'leri
    ref_faces: (F, 3) — üçgen indeksleri (reference ve deformed aynı topoloji)
    def_vertices: (N, 3) — deformed mesh vertex'leri

    return: (F, 3, 3) — her üçgen için 3x3 dönüşüm matrisi S
    """
    V = get_V(ref_vertices, ref_faces)          # (F, 3, 3)
    V_tilde = get_V(def_vertices, ref_faces)    # (F, 3, 3)

    V_inv = np.linalg.inv(V)                    # (F, 3, 3)
    Q = V_tilde @ V_inv                         # (F, 3, 3) — Q = Ṽ V⁻¹

    return Q

