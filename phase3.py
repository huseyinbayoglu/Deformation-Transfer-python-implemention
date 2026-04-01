"""
Phase 3: Correspondence (Section 5)

"Extracts triangle correspondences using the markers provided by the user."


Objective: min wS*ES + wI*EI + wC*EC
  subject to: ṽ_sk = mk  (marker constraints)

ES = smoothness (Eq. 11) — transformations of neighboring triangles should be similar
EI = identity (Eq. 12) — transformations should not deviate much from the identity
EC = closest pt (Eq. 13) — vertices should be close to the target mesh

"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from phase1 import build_edge_to_faces, build_face_adjacency, remove_duplicates
from phase2 import get_V




def compute_face_normals(vertices, faces):
    """Calculate normal vector for all triangles (F, 3)"""
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    normals = np.cross(v2 - v1, v3 - v1)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    # normalizing
    return normals / norms


def compute_vertex_normals(vertices, faces): # ?
    """Calculate normal vector for all vertices (Average of the neighbor normals). (N, 3)"""
    face_normals = compute_face_normals(vertices, faces)
    vertex_normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_normals)
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vertex_normals / norms


def find_closest_valid_points(source_deformed_verts, source_vertex_normals,
                              target_verts, target_vertex_normals):
    """
    For each source vertex, finds the closest valid point on the target mesh (Eq. 13).
    Validity: angle between normals < 90° (dot product > 0).

    return: (N, 3) closest points; invalid ones remain at their original positions.

    """
    tree = cKDTree(target_verts)

    # Find closest 10 candidates and control validity
    k = min(10, len(target_verts))
    dists, idxs = tree.query(source_deformed_verts, k=k)

    n = len(source_deformed_verts)
    # for default values
    closest = source_deformed_verts.copy()

    for i in range(n):
        src_n = source_vertex_normals[i]
        for j in range(k):
            tgt_idx = idxs[i, j]
            tgt_n = target_vertex_normals[tgt_idx]
            if np.dot(src_n, tgt_n) > 0:  # < 90° ns​⋅nt​=∣ns​∣∣nt​∣cos(θ)
                closest[i] = target_verts[tgt_idx]
                break

    return closest






def build_adjacency_pairs(adjacency):
    pairs_i = []
    pairs_j = []
    for i, neighbors in enumerate(adjacency):
        for j in neighbors:
            pairs_i.append(i)
            pairs_j.append(j)
    return np.array(pairs_i, dtype=np.int32), np.array(pairs_j, dtype=np.int32)


def build_system(source_faces, V_inv, adjacency, n_verts, n_faces,
                 wS, wI, wC, marker_dict, closest_points=None):
    """

    Lineer sistem: min ||Ax - b||^2  →  A^T A x = A^T b

    source_faces: (F, 3)
    V_inv: (F, 3, 3)
    adjacency: list of lists
    marker_dict: {source_vertex_idx: target_position (3,)}
    closest_points: (N, 3) veya None

    return: A (sparse), b (3, n_rows)
    """
    # number of columns
    n_unknowns = n_verts + n_faces

    adj_i, adj_j = build_adjacency_pairs(adjacency)
    n_adj_pairs = len(adj_i)

    # Calculate total number of rows
    n_smooth_rows = n_adj_pairs * 3          # ES: her çift × 3 sütun
    n_identity_rows = n_faces * 3            # EI: her üçgen × 3 sütun
    n_closest_rows = n_verts if (wC > 0 and closest_points is not None) else 0
    n_marker_rows = len(marker_dict)
    # Total number of rows
    n_rows = n_smooth_rows + n_identity_rows + n_closest_rows + n_marker_rows

    # (row, col, val) lists for COO format
    # Her ES satırı en fazla 8 entry, her EI satırı 4, EC 1, marker 1
    max_nnz = n_smooth_rows * 8 + n_identity_rows * 4 + n_closest_rows + n_marker_rows
    row_idx = np.zeros(max_nnz, dtype=np.int32)
    col_idx = np.zeros(max_nnz, dtype=np.int32)
    values = np.zeros(max_nnz, dtype=np.float64)
    nnz = 0  # nonzero counter

    # RHS: 3 boyut için
    b = np.zeros((3, n_rows))
    current_row = 0

    # ─── ES: Smoothness (Eq 11) ───
    # ||T_i - T_j||^2_F  for adjacent (i, j)
    #
    # T_i[d,l] = x_{b_i}*G_i[0,l] + x_{c_i}*G_i[1,l] + x_{n+i}*G_i[2,l]
    #          - x_{a_i}*(G_i[0,l] + G_i[1,l] + G_i[2,l])
    #
    # Row: sqrt(wS) * (T_i[d,l] - T_j[d,l]) = 0

    sqrt_wS = np.sqrt(wS)
    for p in range(n_adj_pairs):
        ti = adj_i[p]
        tj = adj_j[p]
        ai, bi, ci = int(source_faces[ti, 0]), int(source_faces[ti, 1]), int(source_faces[ti, 2])
        aj, bj, cj = int(source_faces[tj, 0]), int(source_faces[tj, 1]), int(source_faces[tj, 2])
        Gi = V_inv[ti]
        Gj = V_inv[tj]

        for l in range(3):
            sum_Gi_l = Gi[0, l] + Gi[1, l] + Gi[2, l]
            sum_Gj_l = Gj[0, l] + Gj[1, l] + Gj[2, l]

            # T_i katsayıları (pozitif)
            entries = [
                (ai, -sum_Gi_l * sqrt_wS),
                (bi,  Gi[0, l] * sqrt_wS),
                (ci,  Gi[1, l] * sqrt_wS),
                (n_verts + ti, Gi[2, l] * sqrt_wS),
                # T_j katsayıları (negatif)
                (aj,  sum_Gj_l * sqrt_wS),
                (bj, -Gj[0, l] * sqrt_wS),
                (cj, -Gj[1, l] * sqrt_wS),
                (n_verts + tj, -Gj[2, l] * sqrt_wS),
            ]

            for col, val in entries:
                row_idx[nnz] = current_row
                col_idx[nnz] = col
                values[nnz] = val
                nnz += 1

            # b = 0 (zaten sıfır)
            current_row += 1

    # ─── EI: Identity (Eq 12) ───
    # ||T_i - I||^2_F
    # Row: sqrt(wI) * T_i[d,l] = sqrt(wI) * I[d,l]

    sqrt_wI = np.sqrt(wI)
    for ti in range(n_faces):
        ai, bi, ci = int(source_faces[ti, 0]), int(source_faces[ti, 1]), int(source_faces[ti, 2])
        Gi = V_inv[ti]

        for l in range(3):
            sum_Gi_l = Gi[0, l] + Gi[1, l] + Gi[2, l]

            entries = [
                (ai, -sum_Gi_l * sqrt_wI),
                (bi,  Gi[0, l] * sqrt_wI),
                (ci,  Gi[1, l] * sqrt_wI),
                (n_verts + ti, Gi[2, l] * sqrt_wI),
            ]

            for col, val in entries:
                row_idx[nnz] = current_row
                col_idx[nnz] = col
                values[nnz] = val
                nnz += 1

            # b[d, current_row] = sqrt(wI) * I[d, l]
            for d in range(3):
                b[d, current_row] = sqrt_wI * (1.0 if d == l else 0.0)

            current_row += 1

    # ─── EC: Closest Point (Eq 13) ───
    # ||ṽ_i - c_i||^2
    # Row: sqrt(wC) * x_i = sqrt(wC) * c_i[d]

    if wC > 0 and closest_points is not None:
        sqrt_wC = np.sqrt(wC)
        for vi in range(n_verts):
            row_idx[nnz] = current_row
            col_idx[nnz] = vi
            values[nnz] = sqrt_wC
            nnz += 1

            for d in range(3):
                b[d, current_row] = sqrt_wC * closest_points[vi, d]

            current_row += 1

    # ─── Marker constraints (yüksek ağırlıklı soft constraint) ───
    marker_weight = 1e5
    for s_idx, pos in marker_dict.items():
        row_idx[nnz] = current_row
        col_idx[nnz] = s_idx
        values[nnz] = marker_weight
        nnz += 1

        for d in range(3):
            b[d, current_row] = marker_weight * pos[d]

        current_row += 1

    # Sparse matris oluştur
    A = sp.coo_matrix(
        (values[:nnz], (row_idx[:nnz], col_idx[:nnz])),
        shape=(n_rows, n_unknowns)
    ).tocsc()

    return A, b



# Solving the equation system
def solve_system(A, b, n_verts, n_faces):
    """
    solves normal denklems A^T A x = A^T b  
    LU faktorizasyonu bir kez yapılır, 3 boyut (x,y,z) için backsubstitution.

    return: (n_verts + n_faces, 3) — solved vertex positions
    """
    ATA = (A.T @ A).tocsc()
    lu = splu(ATA)

    result = np.zeros((n_verts + n_faces, 3))
    for d in range(3):
        ATb = A.T @ b[d]
        result[:, d] = lu.solve(ATb)

    return result



# Correspondence
def match_triangles_by_centroid(deformed_source_verts, source_faces,
                                target_verts, target_faces):
    """
    Deforme edilmiş source mesh ile target mesh arasında üçgen
    eşleştirmesi yapar — centroid'leri en yakın olan çiftler eşleşir.

    return: list of (source_tri_idx, target_tri_idx)
    """
    # Source centroid'leri (deforme edilmiş)
    src_centroids = (
        deformed_source_verts[source_faces[:, 0]] +
        deformed_source_verts[source_faces[:, 1]] +
        deformed_source_verts[source_faces[:, 2]]
    ) / 3.0

    # Target centroid'leri
    tgt_centroids = (
        target_verts[target_faces[:, 0]] +
        target_verts[target_faces[:, 1]] +
        target_verts[target_faces[:, 2]]
    ) / 3.0

    tree = cKDTree(tgt_centroids)
    _, nearest_target = tree.query(src_centroids, k=1)

    # Correspondence map: M = {(s1,t1), (s2,t2), ...}
    correspondence = list(zip(range(len(source_faces)), nearest_target.tolist()))
    return correspondence


# ─────────────────────────────────────────────
# Ana fonksiyon: İki fazlı correspondence
# ─────────────────────────────────────────────

def compute_correspondence(source_verts, source_faces,
                           target_verts, target_faces,
                           markers):
    """
    Section 5'teki iki fazlı correspondence algoritması.

    source_verts, source_faces: source reference mesh
    target_verts, target_faces: target reference mesh
    markers: [(source_vertex_idx, target_vertex_idx), ...]

    return: correspondence list [(source_tri, target_tri), ...]
    """
    n_verts = len(source_verts)
    n_faces = len(source_faces)

    # V_inv: source reference mesh'ten (Eq 3-4)
    V = get_V(source_verts, source_faces)
    V_inv = np.linalg.inv(V)

    # Adjacency
    edge_to_faces = build_edge_to_faces(source_faces)
    adjacency = build_face_adjacency(edge_to_faces, n_faces)
    adjacency = remove_duplicates(adjacency)

    # Marker dict: {source_vertex_idx: target_position}
    marker_dict = {}
    for s_idx, t_idx in markers:
        marker_dict[s_idx] = target_verts[t_idx]

    # ═══ FAZ 1: wC = 0 (closest point yok) ═══
    print("Faz 1: wS=1.0, wI=0.001, wC=0")
    A, b = build_system(source_faces, V_inv, adjacency, n_verts, n_faces,
                        wS=1.0, wI=0.001, wC=0.0,
                        marker_dict=marker_dict, closest_points=None)
    deformed = solve_system(A, b, n_verts, n_faces)
    deformed_verts = deformed[:n_verts]
    print(f"  Phase 1 is complete.")

    # ═══ FAZ 2: wC'yi kademeli artır ═══
    target_vnormals = compute_vertex_normals(target_verts, target_faces)
    wC_steps = [1.0, 10.0, 100.0, 500, 1000, 2500,5000.0]

    for step, wC in enumerate(wC_steps):
        print(f"Phase 2 step {step + 1}/{len(wC_steps)}: wC={wC}")

        # Source vertex normal'lerini güncelle (deforme mesh'ten)
        source_vnormals = compute_vertex_normals(deformed_verts, source_faces)

        # En yakın geçerli noktaları bul
        closest = find_closest_valid_points(
            deformed_verts, source_vnormals,
            target_verts, target_vnormals
        )

        # Sistemi yeniden kur ve çöz (her seferinde orijinal source'tan!)
        A, b = build_system(source_faces, V_inv, adjacency, n_verts, n_faces,
                            wS=1.0, wI=0.001, wC=wC,
                            marker_dict=marker_dict, closest_points=closest)
        deformed = solve_system(A, b, n_verts, n_faces)
        deformed_verts = deformed[:n_verts]
        print(f"  Step {step + 1} complete.")

    # ═══ Centroid eşleştirme ═══
    print("Triangle correspondence hesaplanıyor...")
    correspondence = match_triangles_by_centroid(
        deformed_verts, source_faces,
        target_verts, target_faces
    )
    print(f"  {len(correspondence)} triangle matched.")

    return correspondence, deformed_verts
