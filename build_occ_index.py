#!/usr/bin/env python3
"""Build index mapping task names to demo lengths for occ label alignment.

Reads original LIBERO HDF5 files to get per-demo step counts.
Output JSON: {task_name: {occ_file, demos: {demo_key: original_len}}}
"""
import json, os, sys
import h5py

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--libero_dir", type=str, default=None)
    parser.add_argument("--occ_dir", type=str, default=None)
    args = parser.parse_args()
    suite = args.task_suite_name
    libero_dir = args.libero_dir or f"/root/autodl-tmp/libero_datasets/{suite}"
    occ_dir = args.occ_dir or f"/root/occsafe_vla/occ_data/{suite}"
    output = os.path.join(occ_dir, "occ_index.json")

    index = {}
    for fname in sorted(os.listdir(libero_dir)):
        if not fname.endswith(".hdf5") or fname.endswith(".downloading"):
            continue
        task_name = fname.replace("_demo.hdf5", "")
        occ_file = f"occ_{fname}"

        demos = {}
        with h5py.File(os.path.join(libero_dir, fname), "r") as f:
            for key in sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1])):
                demos[key] = int(f[f"data/{key}/actions"].shape[0])

        index[task_name] = {"file": occ_file, "demos": demos}
        print(f"  {task_name}: {len(demos)} demos")

    os.makedirs(occ_dir, exist_ok=True)
    with open(output, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nSaved: {output} ({len(index)} tasks)")

if __name__ == "__main__":
    main()
