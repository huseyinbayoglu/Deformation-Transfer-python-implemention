"""
Deformation Transfer for Triangle Meshes — Main Pipeline
Sumner & Popović, SIGGRAPH 2004

Kullanım:
  python main.py --config source_obj/markers-cat-lion.yml
  python main.py --config source_obj/markers-cat-lion.yml --correspondence corr_cat_lion.npy
"""

import os
import time
import argparse
import numpy as np
import yaml

from phase1 import load_obj, write_obj
from phase2 import compute_source_deformations
from phase3 import compute_correspondence
from phase4 import deformation_transfer



class Timer:
    def __init__(self):
        self.records = []

    def start(self, label):
        self._label = label
        self._t0 = time.time()
        print(f"start: {label}")

    def stop(self):
        elapsed = time.time() - self._t0
        self.records.append((self._label, elapsed))
        print(f"done  {self._label}  ({elapsed:.3f}s)")
        return elapsed

    def summary(self):
        print("\n" + "=" * 50)
        print("ZAMAN ÖZETİ")
        print("=" * 50)
        total = 0
        for label, t in self.records:
            print(f"  {label:<40s} {t:>8.3f}s")
            total += t
        print("-" * 50)
        print(f"  {'TOPLAM':<40s} {total:>8.3f}s")
        print("=" * 50)


# ─────────────────────────────────────────────
# Config okuma
# ─────────────────────────────────────────────

def load_config(config_path):
    base_dir = os.path.dirname(config_path)
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    def resolve(p):
        return os.path.join(base_dir, p)

    config = {
        'source_reference': resolve(cfg['source']['reference']),
        'source_poses': [resolve(p) for p in cfg['source']['poses']],
        'target_reference': resolve(cfg['target']['reference']),
        'markers': [],
    }

    for m in cfg['markers']:
        s, t = m.split(':')
        config['markers'].append((int(s), int(t)))

    return config


# Correspondence I/O

def save_correspondence(correspondence, path):
    arr = np.array(correspondence, dtype=np.int32)
    np.save(path, arr)
    print(f"Correspondence kaydedildi: {path}  ({len(correspondence)} çift)")


def load_correspondence(path):
    arr = np.load(path)
    correspondence = list(map(tuple, arr.tolist()))
    print(f"Correspondence yüklendi: {path}  ({len(correspondence)} çift)")
    return correspondence


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deformation Transfer for Triangle Meshes")
    parser.add_argument('--config', required=True, help='YAML config dosyası (markers-*.yml)')
    parser.add_argument('--correspondence', default=None,
                        help='Önceden hesaplanmış correspondence (.npy). Verilmezse hesaplanır.')
    parser.add_argument('--output-dir', default='output', help='Çıktı klasörü')
    parser.add_argument('--save-correspondence', default=None,
                        help='Hesaplanan correspondence\'ı kaydet (.npy)')
    args = parser.parse_args()

    timer = Timer()
    os.makedirs(args.output_dir, exist_ok=True)

    # ═══ 1. Mesh'leri yükle ═══
    timer.start("Mesh yükleme")
    config = load_config(args.config)

    src_ref_verts, src_ref_faces = load_obj(config['source_reference'])
    tgt_ref_verts, tgt_ref_faces = load_obj(config['target_reference'])

    source_def_verts_list = []
    pose_names = []
    for pose_path in config['source_poses']:
        verts, _ = load_obj(pose_path)
        source_def_verts_list.append(verts)
        pose_names.append(os.path.splitext(os.path.basename(pose_path))[0])

    print(f"  Source: {len(src_ref_verts)} vertices, {len(src_ref_faces)} faces, {len(source_def_verts_list)} poses")
    print(f"  Target: {len(tgt_ref_verts)} vertices, {len(tgt_ref_faces)} faces")
    print(f"  Markers: {len(config['markers'])} çift")
    timer.stop()

    # ═══ 2. Correspondence ═══
    if args.correspondence and os.path.exists(args.correspondence):
        timer.start("Correspondence yükleme (dosyadan)")
        correspondence = load_correspondence(args.correspondence)
        timer.stop()
    else:
        timer.start("Correspondence hesaplama (Phase 3)")
        correspondence, _ = compute_correspondence(
            src_ref_verts, src_ref_faces,
            tgt_ref_verts, tgt_ref_faces,
            config['markers']
        )
        timer.stop()

        # Otomatik kaydet
        save_path = args.save_correspondence
        if save_path is None:
            save_path = os.path.join(args.output_dir, "correspondence.npy")
        save_correspondence(correspondence, save_path)

    # ═══ 3. Deformation Transfer (Phase 4) ═══
    timer.start("Transfer sistemi kurma + LU faktorizasyon")
    from phase4 import build_transfer_system, add_vertex_constraints, factorize
    n_target_verts = len(tgt_ref_verts)
    n_target_faces = len(tgt_ref_faces)

    A, V_inv_target, rhs_src_indices, n_identity_rows = build_transfer_system(
        tgt_ref_verts, tgt_ref_faces, correspondence, n_target_verts
    )
    n_corr_rows = A.shape[0] - n_identity_rows

    # Global translation'ı sabitlemek için en az 1 vertex pin'lenmeli
    # (makale: "unique up to a global translation")
    vertex_constraints = {0: tgt_ref_verts[0]}
    A, _ = add_vertex_constraints(A, A.shape[0], A.shape[1], vertex_constraints)
    lu, A = factorize(A)
    timer.stop()

    # ═══ 4. Her pose'u transfer et ═══
    from phase4 import transfer_single_pose
    for i, def_verts in enumerate(source_def_verts_list):
        timer.start(f"Pose {i+1}/{len(source_def_verts_list)}: {pose_names[i]}")

        # Source deformasyonları
        S = compute_source_deformations(src_ref_verts, src_ref_faces, def_verts)

        # Backsubstitution
        target_deformed = transfer_single_pose(
            lu, A, S, rhs_src_indices,
            n_target_verts, n_target_faces,
            n_corr_rows, n_identity_rows, vertex_constraints
        )

        # OBJ kaydet
        out_path = os.path.join(args.output_dir, f"{pose_names[i]}_transferred.obj")
        write_obj(out_path, target_deformed, tgt_ref_faces)
        print(f"  → {out_path}")
        timer.stop()

    # ═══ Özet ═══
    timer.summary()


if __name__ == "__main__":
    main()

# python3 main.py --config source_obj/markers-horse-camel.yml  
# python3 main.py --config source_obj/markers-cat-lion.yml  