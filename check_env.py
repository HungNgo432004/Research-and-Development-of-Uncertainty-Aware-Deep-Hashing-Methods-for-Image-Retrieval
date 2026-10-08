"""Quick environment check: packages, GPU, pretrained weights, dataset files.

Usage: python check_env.py [--dataset imagenet|coco|nuswide_21]
"""
import argparse
import importlib
import os
import sys


def check_packages():
    ok = True
    for name in ["torch", "torchvision", "numpy", "scipy", "pandas", "sklearn", "matplotlib", "tqdm", "PIL"]:
        try:
            mod = importlib.import_module(name)
            print(f"[OK]   {name:12s} {getattr(mod, '__version__', '')}")
        except ImportError:
            print(f"[MISS] {name}")
            ok = False
    return ok


def check_gpu():
    import torch
    if not torch.cuda.is_available():
        print("[MISS] CUDA not available - training will not run (config uses cuda:0)")
        return False
    props = torch.cuda.get_device_properties(0)
    print(f"[OK]   GPU: {props.name}, {props.total_memory / 1024 ** 3:.1f} GB")
    return True


def check_model():
    import torch
    from network import AlexNet
    net = AlexNet(16).cuda() if torch.cuda.is_available() else AlexNet(16)
    x = torch.randn(2, 3, 224, 224, device=next(net.parameters()).device)
    with torch.no_grad():
        u = net(x)
    print(f"[OK]   AlexNet pretrained weights loaded, output shape {tuple(u.shape)}")
    return True


def check_dataset(dataset):
    root = os.path.join("..", "image_hashing_data", dataset)
    ok = True
    for split in ["train.txt", "new_train.txt", "val.txt", "test.txt", "database.txt"]:
        path = os.path.join(root, split)
        if not os.path.exists(path):
            print(f"[MISS] {path}  (copy from ./data/{dataset}/{split})")
            ok = False
            continue
        lines = open(path).readlines()
        missing = [l.split()[0] for l in lines[:20] if not os.path.exists(os.path.join(root, l.split()[0]))]
        status = "OK  " if not missing else "MISS"
        print(f"[{status}] {path}: {len(lines)} lines" + (f", first 20 entries: {len(missing)} images not found (e.g. {missing[0]})" if missing else ""))
        ok = ok and not missing
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="imagenet")
    args = parser.parse_args()

    print("Python", sys.version.split()[0])
    results = [check_packages(), check_gpu(), check_model(), check_dataset(args.dataset)]
    print("\nAll checks passed." if all(results) else "\nSome checks failed (see [MISS] above).")
