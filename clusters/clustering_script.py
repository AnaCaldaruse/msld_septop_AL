import os
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
import bitbirch.bitbirch as bb

# ============================================================
# Argument parsing
# ============================================================
parser = argparse.ArgumentParser(
    description="Tiered clustering: BitBirch -> Butina subclustering"
)
parser.add_argument("--input",  required=True,
                    help="Input SDF file (all successfully posed compounds)")
parser.add_argument("--outdir", default="clusters",
                    help="Output directory (default: clusters)")

# Fingerprint settings
parser.add_argument("--radius",    type=int,   default=2,
                    help="Morgan fingerprint radius (default: 2)")
parser.add_argument("--nbits",     type=int,   default=2048,
                    help="Morgan fingerprint bit vector size (default: 2048)")
parser.add_argument("--chirality", action="store_true", default=True,
                    help="Include chirality in fingerprints (default: True)")

# BitBirch settings
parser.add_argument("--bb-threshold",        type=float, default=0.97,
                    help="BitBirch diameter threshold (default: 0.97)")
parser.add_argument("--bb-branching-factor", type=int,   default=50,
                    help="BitBirch branching factor (default: 50)")

# Filtering
parser.add_argument("--min-size", type=int, default=2,
                    help="Minimum cluster size to keep (default: 2, keeps clusters > min-size)")

# Butina settings
parser.add_argument("--butina-cutoff", type=float, default=0.50,
                    help="Butina distance cutoff (1 - Tanimoto similarity) (default: 0.50)")

args = parser.parse_args()

# ============================================================
# Setup
# ============================================================
os.makedirs(args.outdir, exist_ok=True)

print(f"\nInput SDF    : {args.input}")
print(f"Output dir   : {args.outdir}")
print(f"FP radius    : {args.radius}  |  nBits: {args.nbits}  |  chirality: {args.chirality}")
print(f"BitBirch     : threshold={args.bb_threshold}, branching_factor={args.bb_branching_factor}")
print(f"Min size     : >{args.min_size} compounds")
print(f"Butina cutoff: {args.butina_cutoff} (similarity >= {1 - args.butina_cutoff:.2f})")

# ============================================================
# Step 1 — Load SDF and compute fingerprints
# ============================================================
print("\n=== STEP 1: Loading SDF & computing fingerprints ===")

gen = AllChem.GetMorganGenerator(
    radius=args.radius,
    fpSize=args.nbits,
    includeChirality=args.chirality
)

supplier = Chem.SDMolSupplier(args.input, removeHs=False)

mols, names, fp_arrays = [], [], []
skipped = 0

for mol in supplier:
    if mol is None:
        skipped += 1
        continue
    try:
        name = mol.GetProp("_Name").strip()
        fp   = gen.GetFingerprint(mol)
        arr  = np.zeros((args.nbits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        mols.append(mol)
        names.append(name)
        fp_arrays.append(arr)
    except Exception as e:
        print(f"  Warning: skipping molecule — {e}")
        skipped += 1

fps_matrix = np.stack(fp_arrays)
print(f"  Loaded  : {len(mols)} compounds  ({skipped} skipped)")
print(f"  FP shape: {fps_matrix.shape}")

# ============================================================
# Step 2 — BitBirch clustering
# ============================================================
print("\n=== STEP 2: BitBirch clustering ===")

bb.set_merge("diameter")
brc = bb.BitBirch(threshold=args.bb_threshold, branching_factor=args.bb_branching_factor)
brc.fit(fps_matrix)

mol_ids  = brc.get_cluster_mol_ids()
clusters = [list(c) for c in mol_ids if len(c) > 0]
sizes    = sorted([len(c) for c in clusters], reverse=True)

print(f"  Total non-empty clusters : {len(clusters)}")
print(f"  Largest cluster          : {sizes[0]}")
print(f"  Avg cluster size         : {np.mean(sizes):.2f}")
print(f"  Singletons (size = 1)    : {sum(1 for s in sizes if s == 1)}")

# ============================================================
# Step 3 — Filter to clusters > min_size
# ============================================================
print(f"\n=== STEP 3: Filtering clusters (> {args.min_size} compounds) ===")

sorted_clusters = sorted(clusters, key=len, reverse=True)

kept = []
for rank, c in enumerate(sorted_clusters, start=1):
    if len(c) > args.min_size:
        kept.append({
            "rank":    rank,
            "indices": c,
            "names":   [names[i] for i in c],
            "mols":    [mols[i]  for i in c],
            "fps":     [gen.GetFingerprint(mols[i]) for i in c],
            "size":    len(c),
        })

print(f"  Clusters kept : {len(kept)}")
print(f"  Total compounds in kept clusters : {sum(cl['size'] for cl in kept)}")

# ============================================================
# Step 4 — Write BitBirch cluster SDFs
# ============================================================
print("\n=== STEP 4: Writing BitBirch cluster SDFs ===")

mol_by_name = {name: mol for name, mol in zip(names, mols)}

for cl in kept:
    filename = os.path.join(args.outdir, f"cluster_{cl['rank']:03d}.sdf")
    writer   = Chem.SDWriter(filename)
    for compound_name in cl["names"]:
        writer.write(mol_by_name[compound_name])
    writer.close()
    print(f"  cluster_{cl['rank']:03d}.sdf  —  {cl['size']} compounds")

# ============================================================
# Step 5 — Butina subclustering over ALL kept clusters
# ============================================================
print("\n=== STEP 5: Butina subclustering ===")

summary_rows        = []
total_subclusters   = 0
total_sub_compounds = 0

for cl in kept:
    sub_fps   = cl["fps"]
    sub_mols  = cl["mols"]
    sub_names = cl["names"]

    # Build pairwise Tanimoto distance matrix
    dists = []
    for i in range(1, len(sub_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(sub_fps[i], sub_fps[:i])
        dists.extend([1 - s for s in sims])

    sub_clusters = Butina.ClusterData(
        dists, len(sub_fps), args.butina_cutoff, isDistData=True
    )
    sub_clusters = sorted(sub_clusters, key=len, reverse=True)

    # Keep subclusters > min_size
    kept_subs = [sc for sc in sub_clusters if len(sc) > args.min_size]

    if not kept_subs:
        print(f"  cluster_{cl['rank']:03d}: no subclusters > {args.min_size} — skipped")
        continue

    sub_dir = os.path.join(args.outdir, f"cluster_{cl['rank']:03d}_refined")
    os.makedirs(sub_dir, exist_ok=True)

    print(f"  cluster_{cl['rank']:03d} ({cl['size']} compounds) → "
          f"{len(kept_subs)} subclusters")

    for sub_rank, sc in enumerate(kept_subs, start=1):
        sub_filename = os.path.join(sub_dir, f"subcluster_{sub_rank:03d}.sdf")
        writer = Chem.SDWriter(sub_filename)
        for local_idx in sc:
            writer.write(sub_mols[local_idx])
        writer.close()

        total_subclusters   += 1
        total_sub_compounds += len(sc)

        print(f"    subcluster_{sub_rank:03d}.sdf  —  {len(sc)} compounds")

        for local_idx in sc:
            summary_rows.append({
                "Name":            sub_names[local_idx],
                "bb_cluster_rank": cl["rank"],
                "bb_cluster_size": cl["size"],
                "subcluster_rank": sub_rank,
                "subcluster_size": len(sc),
            })

# ============================================================
# Step 6 — Save cluster summary CSV
# ============================================================
print("\n=== STEP 6: Saving cluster summary CSV ===")

summary_path = os.path.join(args.outdir, "cluster_summary.csv")
pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
print(f"  Saved → {summary_path}")

# ============================================================
# Final report
# ============================================================
print("\n=== Done ===")
print(f"  BitBirch clusters retained : {len(kept)}")
print(f"  Total subclusters          : {total_subclusters}")
print(f"  Total compounds retained   : {total_sub_compounds}")
print(f"  All files written to       : {args.outdir}/")