"""
Phase 4: Deformation Transfer (Section 4)

Source mesh'in deformasyonunu, correspondence map üzerinden target mesh'e aktarır.

Algoritma:
  1. Source deformasyonlarını hesapla: S_j = Ṽ_j V_j⁻¹  (Eq 4)
  2. Correspondence map ile S_j'leri target üçgenlerine eşle
  3. Target üçgenlerinin dönüşümlerini vertex cinsinden yaz: T_i = Ṽ_i V_i⁻¹
  4. min Σ ||S_sj - T_tj||²_F  çöz  (Eq 8)
     → A^T A x̃ = A^T c   (Eq 9-10)
  5. LU faktorizasyonu bir kez → her yeni pose için sadece backsubstitution
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from phase2 import get_V, compute_source_deformations


def build_transfer_system(target_verts, target_faces, correspondence, n_target_verts):
    """
    Deformation transfer için sparse matris A'yı oluşturur.
    A sadece target mesh'e bağlıdır — bir kez hesaplanır.

    Eşleşmemiş target üçgenlerine küçük ağırlıklı identity terimi (T_i ≈ I)
    eklenir, böylece tüm sütunlar kapsanır ve AᵀA non-singular olur.

    target_verts: (N_t, 3) — target reference vertex'leri
    target_faces: (F_t, 3) — target üçgen indeksleri
    correspondence: [(source_tri, target_tri), ...]
    n_target_verts: int

    return: A (sparse), V_inv_target, rhs_source_tri_indices, n_identity_rows
    """
    n_target_faces = len(target_faces)
    n_unknowns = n_target_verts + n_target_faces

    # Target mesh için V_inv hesapla
    V_target = get_V(target_verts, target_faces)     # (F_t, 3, 3)
    V_inv_target = np.linalg.inv(V_target)           # (F_t, 3, 3)

    # Hangi target üçgenleri eşleşmemiş?
    mapped_tgt = set(tgt for _, tgt in correspondence)
    unmapped_tgt = [t for t in range(n_target_faces) if t not in mapped_tgt]

    n_pairs = len(correspondence)
    n_unmapped = len(unmapped_tgt)

    # Satır sayıları: correspondence (3/pair) + unmapped identity (3/tri)
    n_corr_rows = n_pairs * 3
    n_identity_rows = n_unmapped * 3
    n_total_rows = n_corr_rows + n_identity_rows

    max_nnz = n_corr_rows * 4 + n_identity_rows * 4
    row_idx = np.zeros(max_nnz, dtype=np.int32)
    col_idx = np.zeros(max_nnz, dtype=np.int32)
    values = np.zeros(max_nnz, dtype=np.float64)
    nnz = 0
    current_row = 0

    rhs_source_tri_indices = []

    # ─── Correspondence terimleri: ||S_s - T_t||²_F ───
    for src_tri, tgt_tri in correspondence:
        a, b_v, c = (int(target_faces[tgt_tri, 0]),
                      int(target_faces[tgt_tri, 1]),
                      int(target_faces[tgt_tri, 2]))
        G = V_inv_target[tgt_tri]

        for l in range(3):
            sum_G_l = G[0, l] + G[1, l] + G[2, l]

            entries = [
                (a,                        -sum_G_l),
                (b_v,                       G[0, l]),
                (c,                         G[1, l]),
                (n_target_verts + tgt_tri,  G[2, l]),
            ]

            for col_i, val in entries:
                row_idx[nnz] = current_row
                col_idx[nnz] = col_i
                values[nnz] = val
                nnz += 1

            current_row += 1

        rhs_source_tri_indices.append(src_tri)

    # ─── Eşleşmemiş üçgenler için identity terimi: wI * ||T_i - I||²_F ───
    # Küçük ağırlık: bu üçgenlerin şeklini korur, sistemi non-singular yapar
    wI_unmapped = 0.001
    sqrt_wI = np.sqrt(wI_unmapped)

    for tgt_tri in unmapped_tgt:
        a, b_v, c = (int(target_faces[tgt_tri, 0]),
                      int(target_faces[tgt_tri, 1]),
                      int(target_faces[tgt_tri, 2]))
        G = V_inv_target[tgt_tri]

        for l in range(3):
            sum_G_l = G[0, l] + G[1, l] + G[2, l]

            entries = [
                (a,                        -sum_G_l * sqrt_wI),
                (b_v,                       G[0, l] * sqrt_wI),
                (c,                         G[1, l] * sqrt_wI),
                (n_target_verts + tgt_tri,  G[2, l] * sqrt_wI),
            ]

            for col_i, val in entries:
                row_idx[nnz] = current_row
                col_idx[nnz] = col_i
                values[nnz] = val
                nnz += 1

            current_row += 1

    print(f"  Correspondence: {n_pairs} çift, Eşleşmemiş: {n_unmapped} üçgen")

    A = sp.coo_matrix(
        (values[:nnz], (row_idx[:nnz], col_idx[:nnz])),
        shape=(n_total_rows, n_unknowns)
    ).tocsc()

    return A, V_inv_target, rhs_source_tri_indices, n_identity_rows


def add_vertex_constraints(A, n_rows, n_unknowns, constraints):
    """
    Vertex pozisyon kısıtları ekler (örn. ayak sabitleme).

    constraints: {vertex_idx: position (3,)}

    return: A_new (sparse, ek satırlar eklenmiş)
    """
    if not constraints:
        return A, n_rows

    constraint_weight = 1e5
    n_constraints = len(constraints)

    # Ek satırlar
    extra_rows = []
    extra_cols = []
    extra_vals = []

    for i, (v_idx, _) in enumerate(constraints.items()):
        extra_rows.append(i)
        extra_cols.append(v_idx)
        extra_vals.append(constraint_weight)

    extra = sp.coo_matrix(
        (extra_vals, (extra_rows, extra_cols)),
        shape=(n_constraints, A.shape[1])
    ).tocsc()

    A_new = sp.vstack([A, extra]).tocsc()
    return A_new, n_rows + n_constraints


def factorize(A):
    """A^T A'yı LU faktörize eder. Bir kez çalışır."""
    ATA = (A.T @ A).tocsc()
    return splu(ATA), A


def build_rhs_for_pose(source_deformations, rhs_source_tri_indices, dim,
                       n_corr_rows, n_identity_rows, constraints=None):
    """
    Belirli bir source pose ve boyut (x/y/z) için rhs vektörü b'yi oluşturur.

    source_deformations: (F_s, 3, 3) — S matrislerinin non-translational kısmı
    rhs_source_tri_indices: correspondence'tan gelen source triangle indeksleri
    dim: 0, 1, veya 2 (x, y, z)
    n_corr_rows: correspondence satır sayısı
    n_identity_rows: unmapped identity satır sayısı

    return: b vektörü
    """
    n_constraint_rows = len(constraints) if constraints else 0
    total_rows = n_corr_rows + n_identity_rows + n_constraint_rows
    b = np.zeros(total_rows)

    # Correspondence satırları: b = S[dim, l]
    for i, src_tri in enumerate(rhs_source_tri_indices):
        S = source_deformations[src_tri]  # 3x3
        for l in range(3):
            b[i * 3 + l] = S[dim, l]

    # Identity satırları: b = sqrt(wI) * I[dim, l]
    sqrt_wI = np.sqrt(0.001)
    n_unmapped = n_identity_rows // 3
    for i in range(n_unmapped):
        for l in range(3):
            b[n_corr_rows + i * 3 + l] = sqrt_wI * (1.0 if dim == l else 0.0)

    # Vertex constraint'ler
    if constraints:
        constraint_weight = 1e5
        for j, (v_idx, pos) in enumerate(constraints.items()):
            b[n_corr_rows + n_identity_rows + j] = constraint_weight * pos[dim]

    return b


def transfer_single_pose(lu, A, source_deformations, rhs_source_tri_indices,
                         n_target_verts, n_target_faces,
                         n_corr_rows, n_identity_rows, constraints=None):
    """
    Tek bir source pose'u target mesh'e aktarır (sadece backsubstitution).

    return: (n_target_verts, 3) — deformed target vertex pozisyonları
    """
    result = np.zeros((n_target_verts + n_target_faces, 3))

    for d in range(3):
        b = build_rhs_for_pose(source_deformations, rhs_source_tri_indices,
                               d, n_corr_rows, n_identity_rows, constraints)
        ATb = A.T @ b
        result[:, d] = lu.solve(ATb)

    return result[:n_target_verts]


def deformation_transfer(source_ref_verts, source_ref_faces,
                         source_def_verts_list,
                         target_ref_verts, target_faces,
                         correspondence,
                         vertex_constraints=None):
    """
    Deformation Transfer ana fonksiyonu.

    source_ref_verts: (N_s, 3) — source reference mesh
    source_ref_faces: (F_s, 3) — source faces
    source_def_verts_list: list of (N_s, 3) — source deformed pose'lar
    target_ref_verts: (N_t, 3) — target reference mesh
    target_faces: (F_t, 3) — target faces
    correspondence: [(source_tri, target_tri), ...]
    vertex_constraints: {vertex_idx: position (3,)} veya None

    return: list of (N_t, 3) — her pose için deformed target vertex'leri
    """
    n_target_verts = len(target_ref_verts)
    n_target_faces = len(target_faces)

    if vertex_constraints is None:
        vertex_constraints = {}

    # 1. Transfer sistemi kur (target mesh'e bağlı, bir kez)
    print("Transfer sistemi kuruluyor...")
    A, V_inv_target, rhs_src_indices, n_identity_rows = build_transfer_system(
        target_ref_verts, target_faces, correspondence, n_target_verts
    )
    n_corr_rows = A.shape[0] - n_identity_rows

    # 2. Vertex constraint'leri ekle
    if not vertex_constraints:
        vertex_constraints = {0: target_ref_verts[0]}
    A, total_rows = add_vertex_constraints(
        A, A.shape[0], A.shape[1], vertex_constraints
    )

    # 3. LU faktorizasyonu (bir kez!)
    print("LU faktorizasyonu...")
    lu, A = factorize(A)
    print("  Faktorizasyon tamamlandı.")

    # 4. Her pose için backsubstitution
    results = []
    for i, def_verts in enumerate(source_def_verts_list):
        print(f"Pose {i + 1}/{len(source_def_verts_list)} transfer ediliyor...")

        # Source deformasyonlarını hesapla: S = Ṽ V⁻¹
        S = compute_source_deformations(source_ref_verts, source_ref_faces, def_verts)

        # Backsubstitution ile çöz
        target_deformed = transfer_single_pose(
            lu, A, S, rhs_src_indices,
            n_target_verts, n_target_faces,
            n_corr_rows, n_identity_rows, vertex_constraints
        )
        results.append(target_deformed)

    return results
