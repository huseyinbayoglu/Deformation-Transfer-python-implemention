"""
Deformation Transfer sonuçlarını 3D görselleştirir.

"""

import os
import argparse
import numpy as np
import pyvista as pv

from phase1 import load_obj


def mesh_to_pyvista(vertices, faces):
    """numpy vertices/faces → pyvista PolyData"""
    n_faces = len(faces)
    pv_faces = np.column_stack([np.full(n_faces, 3), faces]).ravel()
    return pv.PolyData(vertices, pv_faces)


def load_config_paths(config_path):
    import yaml
    base_dir = os.path.dirname(config_path)
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    def resolve(p):
        return os.path.join(base_dir, p)

    result = {
        'source_reference': resolve(cfg['source']['reference']),
        'source_poses': [resolve(p) for p in cfg['source']['poses']],
        'target_reference': resolve(cfg['target']['reference']),
        'target_poses': [],
    }
    if 'poses' in cfg.get('target', {}):
        result['target_poses'] = [resolve(p) for p in cfg['target']['poses']]
    return result


def show_comparison(source_ref_path, source_pose_path, target_ref_path, transferred_path, title=""):
    """2x2 grid: source ref/pose (üst), target ref/transferred (alt)"""
    src_ref_v, src_ref_f = load_obj(source_ref_path)
    src_pose_v, _ = load_obj(source_pose_path)
    tgt_ref_v, tgt_ref_f = load_obj(target_ref_path)
    tgt_trans_v, _ = load_obj(transferred_path)

    pl = pv.Plotter(shape=(2, 2), window_size=(1400, 1000))

    pl.subplot(0, 0)
    pl.add_mesh(mesh_to_pyvista(src_ref_v, src_ref_f), color='lightblue', show_edges=False)
    pl.add_text("Source Reference", font_size=10)

    pl.subplot(0, 1)
    pl.add_mesh(mesh_to_pyvista(src_pose_v, src_ref_f), color='steelblue', show_edges=False)
    pl.add_text("Source Pose", font_size=10)

    pl.subplot(1, 0)
    pl.add_mesh(mesh_to_pyvista(tgt_ref_v, tgt_ref_f), color='lightyellow', show_edges=False)
    pl.add_text("Target Reference", font_size=10)

    pl.subplot(1, 1)
    pl.add_mesh(mesh_to_pyvista(tgt_trans_v, tgt_ref_f), color='orange', show_edges=False)
    pl.add_text("Transferred Result", font_size=10)

    if title:
        pl.add_text(title, position='upper_edge', font_size=12)

    pl.link_views()
    pl.show()


def show_all_poses(source_ref_path, source_poses, target_ref_path, tgt_ref_faces,
                   transferred_paths, target_gt_poses=None):
    """
    3 satırlı grid:
      Satır 1 (mavi): Source reference + source pose'lar
      Satır 2 (turuncu): Target reference + transferred sonuçlar
      Satır 3 (yeşil): Target reference + ground truth pose'lar (varsa)
    """
    n = len(transferred_paths)
    show_n = min(n, 9)
    n_cols = show_n + 1  # reference + pose'lar 

    n_rows = 2

    pl = pv.Plotter(shape=(n_rows, n_cols), window_size=(2900, 750 * n_rows))

    src_ref_v, src_ref_f = load_obj(source_ref_path)
    tgt_ref_v, _ = load_obj(target_ref_path)

    # ─── Satır 1: Source (mavi) ───
    pl.subplot(0, 0)
    pl.add_mesh(mesh_to_pyvista(src_ref_v, src_ref_f), color='lightblue', show_edges=False)
    pl.add_text("Source Ref", font_size=9)

    for i in range(show_n):
        src_v, _ = load_obj(source_poses[i])
        pl.subplot(0, i + 1)
        pl.add_mesh(mesh_to_pyvista(src_v, src_ref_f), color='steelblue', show_edges=False)
        pl.add_text(f"Src Pose {i+1}", font_size=9)

    # ─── Satır 2: Transferred (turuncu) ───
    pl.subplot(1, 0)
    pl.add_mesh(mesh_to_pyvista(tgt_ref_v, tgt_ref_faces), color='lightyellow', show_edges=False)
    pl.add_text("Target Ref", font_size=9)

    for i in range(show_n):
        tgt_v, _ = load_obj(transferred_paths[i])
        pl.subplot(1, i + 1)
        pl.add_mesh(mesh_to_pyvista(tgt_v, tgt_ref_faces), color='orange', show_edges=False)
        pl.add_text(f"Transfer {i+1}", font_size=9)

    pl.link_views()
    pl.show()


def main():
    parser = argparse.ArgumentParser(description="Deformation Transfer 3D Görselleştirme")
    parser.add_argument('--config', required=True, help='YAML config dosyası')
    parser.add_argument('--output-dir', default='output', help='Transfer çıktı klasörü')
    parser.add_argument('--pose', type=int, default=None, help='Tek pose göster (1-based index)')
    parser.add_argument('--all', action='store_true', help='Tüm pose\'ları grid halinde göster')
    args = parser.parse_args()

    paths = load_config_paths(args.config)
    _, tgt_ref_f = load_obj(paths['target_reference'])

    # Transferred OBJ dosyalarını bul
    pose_names = [os.path.splitext(os.path.basename(p))[0] for p in paths['source_poses']]
    transferred_paths = [os.path.join(args.output_dir, f"{name}_transferred.obj") for name in pose_names]

    existing = [(i, p) for i, p in enumerate(transferred_paths) if os.path.exists(p)]
    if not existing:
        print(f"Hata: {args.output_dir}/ içinde transferred OBJ bulunamadı.")
        return

    print(f"{len(existing)} transferred pose bulundu.")

    # Ground truth target pose'ları (varsa)
    target_gt = paths.get('target_poses', [])
    target_gt_existing = [p for p in target_gt if os.path.exists(p)] or None

    if args.all:
        show_all_poses(
            paths['source_reference'],
            paths['source_poses'],
            paths['target_reference'],
            tgt_ref_f,
            [p for _, p in existing],
            target_gt_poses=target_gt_existing
        )
    elif args.pose is not None:
        idx = args.pose - 1
        if idx < 0 or idx >= len(transferred_paths):
            print(f"Hata: --pose {args.pose} geçersiz. 1-{len(transferred_paths)} arası olmalı.")
            return
        if not os.path.exists(transferred_paths[idx]):
            print(f"Hata: {transferred_paths[idx]} bulunamadı.")
            return
        show_comparison(
            paths['source_reference'],
            paths['source_poses'][idx],
            paths['target_reference'],
            transferred_paths[idx],
            title=f"Pose {args.pose}: {pose_names[idx]}"
        )
    else:
        # Default: ilk pose'u göster
        idx, path = existing[0]
        show_comparison(
            paths['source_reference'],
            paths['source_poses'][idx],
            paths['target_reference'],
            path,
            title=f"Pose {idx+1}: {pose_names[idx]}"
        )


if __name__ == "__main__":
    main()

# python3 visualize.py --config source_obj/markers-horse-camel.yml --all
# python3 visualize.py --config source_obj/markers-cat-lion.yml --all