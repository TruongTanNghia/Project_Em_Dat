"""Wire Kaggle BUSI breast ultrasound samples into the test pipeline.

WORKFLOW:
    1. Tải BUSI từ Kaggle (không cần đăng nhập nếu dùng kaggle CLI có
       token):
       https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

    2. Giải nén → thường ra thư mục:
           Dataset_BUSI_with_GT/
               benign/    benign (1).png, benign (1)_mask.png, ...
               malignant/ malignant (1).png, ...
               normal/    normal (1).png, ...

    3. Chạy:
           python frontend-next/scripts/wire-busi-samples.py <PATH_TO_BUSI>

    4. (Optional) batch-test endpoint /api/predict-breast:
           python frontend-next/scripts/wire-busi-samples.py <PATH> --api http://localhost:5000

OUTPUT: test_images/busi_<label>_<NN>.png + masks + manifest.json
"""
import argparse
import json
import random
import shutil
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


CLASSES = ['benign', 'malignant', 'normal']


def find_class_dirs(busi_root):
    """Locate benign/ malignant/ normal/ subfolders. Handle nested structures."""
    busi_root = Path(busi_root)
    candidates = [
        busi_root,
        busi_root / 'Dataset_BUSI_with_GT',
        busi_root / 'BUSI',
        busi_root / 'Dataset_BUSI',
    ]
    for c in candidates:
        if (c / 'benign').is_dir() and (c / 'malignant').is_dir():
            print(f"[OK] Tim thay BUSI folders trong: {c}")
            return {
                'benign':    c / 'benign',
                'malignant': c / 'malignant',
                'normal':    c / 'normal' if (c / 'normal').is_dir() else None,
            }
    raise SystemExit(
        f"[FAIL] Khong tim thay benign/ + malignant/ subfolder.\n"
        f"Searched: {[str(c) for c in candidates]}\n"
        f"Kiem tra xem anh da extract zip xong chua?"
    )


def pick_samples(class_dirs, per_class, seed=42):
    """Pick N random samples per class (excluding _mask files)."""
    random.seed(seed)
    picks = []
    for label, d in class_dirs.items():
        if d is None:
            print(f"[WARN] Class '{label}' khong ton tai, skip")
            continue
        all_files = sorted([
            p for p in d.glob('*.png')
            if '_mask' not in p.stem.lower()
        ])
        if not all_files:
            print(f"[WARN] Class '{label}' rong")
            continue
        n = min(per_class, len(all_files))
        sample = random.sample(all_files, n)
        for s in sample:
            mask = s.parent / f"{s.stem}_mask.png"
            picks.append({
                'label': label,
                'image': s,
                'mask':  mask if mask.exists() else None,
            })
        print(f"  [{label:10}] {n}/{len(all_files)} samples")
    return picks


def copy_to_test_dir(picks, out_dir):
    """Copy picked samples to flat test directory with descriptive names."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Wipe previous BUSI samples (keep synthetic ones)
    removed = 0
    for old in out_dir.glob('busi_*'):
        old.unlink()
        removed += 1
    if removed:
        print(f"[CLEAN] Xoa {removed} BUSI file cu")

    # Number per class
    label_counter = {c: 0 for c in CLASSES}
    manifest = []
    for pick in picks:
        label = pick['label']
        label_counter[label] += 1
        idx = label_counter[label]
        new_name = f"busi_{label}_{idx:02d}.png"
        new_path = out_dir / new_name
        shutil.copy2(pick['image'], new_path)

        mask_name = None
        if pick['mask']:
            mask_name = f"busi_{label}_{idx:02d}_mask.png"
            shutil.copy2(pick['mask'], out_dir / mask_name)

        manifest.append({
            'label':  label,
            'file':   new_name,
            'mask':   mask_name,
            'source': str(pick['image']).replace('\\', '/'),
        })
        size_kb = new_path.stat().st_size // 1024
        print(f"  [{label:10}] {new_name}  ({size_kb} KB)")

    manifest_path = out_dir / 'busi_manifest.json'
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    print(f"\n[MANIFEST] {manifest_path}")
    return manifest


def batch_test_api(manifest, out_dir, api_url):
    """POST each sample to /api/predict-breast + save responses."""
    if not HAS_REQUESTS:
        print("[WARN] pip install requests de dung --api flag")
        return

    results_dir = Path(out_dir) / 'busi_results'
    results_dir.mkdir(exist_ok=True)
    print(f"\n[BATCH TEST] {len(manifest)} samples -> {api_url}/api/predict-breast\n")

    all_results = []
    for entry in manifest:
        new_name = entry['file']
        image_path = Path(out_dir) / new_name
        try:
            with open(image_path, 'rb') as f:
                r = requests.post(
                    f"{api_url}/api/predict-breast",
                    files={'image': (new_name, f, 'image/png')},
                    timeout=60,
                )
            data = r.json() if r.headers.get('content-type', '').startswith('application/json') else {}
            verdict = '[OK]' if r.status_code == 200 else '[FAIL]'
            birads = data.get('birads', data.get('fallback_mock', {}).get('birads', '?'))
            mock = data.get('mock', data.get('fallback_mock', {}).get('mock', None))
            mode = 'mock' if mock else ('real' if mock is False else '?')
            print(f"  {verdict:6} [{entry['label']:10}] {new_name:30}  status={r.status_code}  birads={birads}  mode={mode}")

            (results_dir / f"{new_name}.json").write_text(
                json.dumps({
                    'sample':     new_name,
                    'true_label': entry['label'],
                    'status':     r.status_code,
                    'response':   data,
                }, indent=2, ensure_ascii=False),
                encoding='utf-8',
            )
            all_results.append({
                'sample':     new_name,
                'true_label': entry['label'],
                'birads':     birads,
                'mode':       mode,
            })
        except Exception as e:
            print(f"  [ERR]  [{entry['label']:10}] {new_name}: {type(e).__name__}: {e}")
            all_results.append({
                'sample':     new_name,
                'true_label': entry['label'],
                'error':      str(e),
            })

    summary_path = results_dir / 'summary.json'
    summary_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    print(f"\n[SUMMARY] {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Wire Kaggle BUSI samples into test pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example:\n  python wire-busi-samples.py "C:\\\\downloads\\\\Dataset_BUSI_with_GT" --per-class 3 --api http://localhost:5000',
    )
    parser.add_argument('busi_dir', type=str,
                        help='Path to extracted BUSI folder')
    parser.add_argument('--per-class', type=int, default=3,
                        help='Samples per class (default: 3, so 9 total)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output dir (default: <project_root>/test_images)')
    parser.add_argument('--api', type=str, default=None,
                        help='API base URL for batch test (e.g., http://localhost:5000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    out_dir = Path(args.out) if args.out else project_root / 'test_images'

    print("=" * 60)
    print("BUSI Sample Wiring")
    print("=" * 60)
    print(f"Source:    {args.busi_dir}")
    print(f"Output:    {out_dir}")
    print(f"Per class: {args.per_class}")
    print(f"Seed:      {args.seed}")
    if args.api:
        print(f"API:       {args.api}")
    print()

    print("[1/3] Locating BUSI folders...")
    class_dirs = find_class_dirs(args.busi_dir)

    print("\n[2/3] Picking samples...")
    picks = pick_samples(class_dirs, args.per_class, seed=args.seed)
    if not picks:
        raise SystemExit("[FAIL] Khong pick duoc sample nao")

    print("\n[3/3] Copying to test directory...")
    manifest = copy_to_test_dir(picks, out_dir)

    if args.api:
        batch_test_api(manifest, out_dir, args.api.rstrip('/'))

    print()
    print("=" * 60)
    print("DONE.")
    print(f"  {len(manifest)} sample copied to: {out_dir}")
    print(f"  Upload bat ky file busi_*.png nao vao tab 'Ung thu vu' de test")
    print("=" * 60)


if __name__ == '__main__':
    main()
