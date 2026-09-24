# Faster substructure search with roaring bitmaps and an inverted index

One of the most insightful talks at RDKit UGM 2026 was Maciej Wójcikowski's talk about using roaring bitmaps to speed up substructure searches. Having some time in the evening after the UGM and during the hackathon, I decided to take a closer look.

Imagine you have a large corpus of SMARTS substructures (>50k) and want to find which ones are present in a molecule. Maciej talked about this approach being used in Merck's Synthia retrosynthesis software, which makes sense if you have reaction templates and want to know which ones match the starting molecule. For my benchmark, I used the `uspto-grants` subset of `ord-data`: 1.77 million chemical reactions mined from United States patent grants. These reactions are also atom-mapped, so we can use AstraZeneca's `reaction-utils` library to get reasonably good templates. I sampled 50,000 unique templates as substructures and 1,000 unique reactants for the searches.

![Example of matches in the data used](artifacts/reactant_ord_b3ca7045_matches.svg)

## Naive approach

Like Maciej, I started with a naive approach: test each molecule against each query in a loop. Not very creative, but it works.

```python
matches = []
for r_smiles, r_mol in tqdm.tqdm(reactant_rows):
    for s_smiles, s_mol in zip(sub_smiles, sub_mols):
        if r_mol.HasSubstructMatch(s_mol):
            matches.append((r_smiles, s_smiles))
```

**76.307 seconds**—probably fine for a one-time search, but is it sustainable at scale?

## PatternFingerprint

In the next iteration, I tried the next thing suggested by Maciej (and the RDKit documentation): pattern fingerprints. This binary fingerprint uses a set of generic patterns. A query can match a molecule only if the molecule's fingerprint is a superset of the query fingerprint.

```python
...
def to_pattern(mol):
    fp = PatternFingerprint(mol)
    return fp
...

for r_smiles, r_mol in tqdm.tqdm(reactant_rows):
    r_fp = to_pattern(r_mol)
    for s_smiles, s_mol, s_fp in zip(sub_smiles, sub_mols, sub_pats):
        if not AllProbeBitsMatch(s_fp, r_fp):
            continue
        if r_mol.HasSubstructMatch(s_mol):
            matches.append((r_smiles, s_smiles))
```
**22.67 seconds**—3.37× faster than the original version. Bit vector comparison is fast and lets us skip expensive substructure matches.

## Roaring bitmaps

Here comes the main topic: roaring bitmaps. At the UGM, only Andrew Daike raised his hand when asked whether anyone had experience with them. Apparently, more exotic data structures are not that common in the cheminformatics world.

Roaring bitmaps are supposed to be more memory-efficient than standard bitmaps (`SparseBitVect` in RDKit terms), with faster AND, OR, and XOR operations—and that's exactly what we want. In the simplest case, all it takes is installing `pyroaring` and converting the original fingerprint to a bitmap.

```python
from pyroaring import BitMap
...
def to_pattern(mol):
    fp = PatternFingerprint(mol)
    return BitMap(fp.GetOnBits())
...
matches = []
for r_smiles, r_mol in tqdm.tqdm(reactant_rows):
    r_fp = to_pattern(r_mol)
    for s_smiles, s_mol, s_fp in zip(sub_smiles, sub_mols, sub_pats):
        if not s_fp.issubset(r_fp):
            continue
        if r_mol.HasSubstructMatch(s_mol):
            matches.append((r_smiles, s_smiles))
```

**5.823 seconds**—3.89× faster than the previous version and 13.1× faster than the initial one. Impressive, especially for a three-line change.

## Postgres 18 + RDKit cartridge

To complete the set of approaches from Maciej's talk, I tried the Postgres cartridge (according to the RDKit survey, 20% of users use it—can you believe it?). This version consists of loading the queries into a column of the Query Mol (`qmol`) type, adding a GiST index, and then searching one molecule at a time.

For a fair comparison, the reference Postgres time comes from a container with a single CPU and a single connection (because everything can be made faster with multiprocessing, but where's the fun?). The result was **8.235 seconds**.

Adding more CPUs alone does not make the search faster. For the 50k queries used here, the Postgres planner refused to start multiple workers for `Parallel Bitmap Heap Scan`, so both the index scan and recheck were sequential. But using 4 CPUs and 10 connections brought the search down to **1.6 seconds**—the fastest so far, though *slightly cheating* because we're entering multiprocessing territory.

## Roaring postings list

To extend the benchmark, I decided to build another solution on top of roaring bitmaps: a postings list (also known as an inverted index). Postings lists in search engines map content (such as words) to the documents that contain it.

In our case, the content is a bit in the pattern fingerprint, and the documents are the queries that have that bit. We can use a roaring bitmap to store each set of queries. With this structure, the search follows these steps:

1. Build the postings lists from all the queries.
1. Sort the pattern fingerprint bits by frequency.
1. Check the K most frequent bits from the queries against the target molecule.
1. Immediately skip groups of queries whose bit is missing from the target.
1. Use the original roaring bitmap search for what remains.

Time: **1.610 seconds**.

The reference index implementation is in `benchmarks/query_index.py`.

## NumPy version

Lastly, I wanted to check whether a NumPy implementation could compete with roaring bitmaps. Codex/GPT-5.6-Sol helped me a lot with this. We ended up with a solution that stores the packed parts of the original 2,048-bit fingerprint in a set of 32 `int64` values and uses a combination of clever bitwise operations and bit shifts to produce results even faster: **1.261 seconds**.

Is it clever? Probably.
Was it a NumPy learning experience for me? Definitely.
Would I like to support this code? Probably not. The roaring bitmap version is much more intuitive and readable.

![alt text](benchmarks/plots/all_methods_50k.png)
![alt text](benchmarks/plots/postings_comparison.png)
