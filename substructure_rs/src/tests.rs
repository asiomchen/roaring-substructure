//! Unit tests. Expected values were taken from RDKit 2026.03.6
//! (`Chem.MolFromSmiles`, `GetRingInfo().AtomRings()`, `HasSubstructMatch`).
//!
//! Author: Marcin Kowiel + Claude

use crate::fingerprint::{query_fp, target_fp};
use crate::index::Index;
use crate::matcher::{has_match, Plan, Target};
use crate::mol::{parse_smiles, BondType, Mol};
use crate::rings::symmetrized_sssr;
use crate::smarts::parse_smarts;

/// (atomic number, aromatic, total H, degree, charge) per atom.
fn atoms(smiles: &str) -> Vec<(u8, bool, u8, u8, i8)> {
    parse_smiles(smiles)
        .unwrap()
        .target_atoms()
        .iter()
        .map(|a| (a.z, a.aromatic, a.total_h, a.degree, a.charge))
        .collect()
}

fn bonds(smiles: &str) -> Vec<BondType> {
    parse_smiles(smiles).unwrap().bonds.iter().map(|b| b.kind).collect()
}

fn matches(smarts: &str, smiles: &str) -> bool {
    let q = parse_smarts(smarts).unwrap();
    has_match(&q, &Plan::new(&q), &Target::new(&parse_smiles(smiles).unwrap()))
}

fn rings(smiles: &str) -> Vec<Vec<usize>> {
    let mol: Mol = parse_smiles(smiles).unwrap();
    symmetrized_sssr(&mol).atom_rings
}

// ------------------------------------------------------------ sanitization

#[test]
fn kekule_benzene_becomes_aromatic() {
    assert_eq!(atoms("C1=CC=CC=C1"), vec![(6, true, 1, 2, 0); 6]);
    assert_eq!(bonds("C1=CC=CC=C1"), vec![BondType::Aromatic; 6]);
}

#[test]
fn pyridine_and_pyrrole_hydrogens() {
    let pyridine = atoms("c1ccncc1");
    assert_eq!(pyridine[3], (7, true, 0, 2, 0));
    assert_eq!(pyridine[0], (6, true, 1, 2, 0));
    assert_eq!(atoms("c1cc[nH]c1")[3], (7, true, 1, 2, 0));
}

#[test]
fn pyridone_with_exocyclic_carbonyl_is_aromatic() {
    let a = atoms("O=c1cc[nH]cc1");
    assert_eq!(a[0], (8, false, 0, 1, 0));
    assert!(a[1..].iter().all(|x| x.1));
    assert_eq!(bonds("O=c1cc[nH]cc1")[0], BondType::Double);
}

#[test]
fn aryl_radical_keeps_zero_hydrogens() {
    assert_eq!(atoms("[c]1ccccc1")[0], (6, true, 0, 2, 0));
}

#[test]
fn nitro_group_is_charge_separated() {
    assert_eq!(
        atoms("CN(=O)=O"),
        vec![(6, false, 3, 1, 0), (7, false, 0, 3, 1), (8, false, 0, 1, -1), (8, false, 0, 1, 0)]
    );
    assert_eq!(atoms("C[N+](=O)[O-]")[3], (8, false, 0, 1, -1));
}

#[test]
fn perchlorate_is_charge_separated() {
    let a = atoms("OCl(=O)(=O)=O");
    assert_eq!(a[1], (17, false, 0, 4, 3));
    assert!(a[2..].iter().all(|x| x.4 == -1));
}

#[test]
fn stereo_defining_hydrogen_is_kept() {
    let a = atoms("[H]/N=C(/C)N");
    assert_eq!(a.len(), 5);
    assert_eq!(a[0], (1, false, 0, 1, 0));
    assert_eq!(a[1], (7, false, 1, 2, 0));
}

#[test]
fn plain_explicit_hydrogen_is_removed() {
    assert_eq!(atoms("[H]C"), vec![(6, false, 4, 0, 0)]);
}

#[test]
fn dative_bond_parses() {
    assert_eq!(atoms("[NH3]->[Pt]"), vec![(7, false, 3, 1, 0), (78, false, 0, 1, 0)]);
    assert_eq!(bonds("[NH3]->[Pt]"), vec![BondType::Dative]);
}

#[test]
fn aryne_keeps_aromatic_triple_bond() {
    let a = atoms("COc1c#cccc1");
    assert!(a[2..].iter().all(|x| x.1));
    assert_eq!(bonds("COc1c#cccc1")[3], BondType::Triple);
}

#[test]
fn furan_in_macrocycle_is_not_aromatic() {
    // The furan O is first seen in the 11-membered ring, where RDKit rules it out.
    let a = atoms("C=C(C)[C@@H]1CCC2=C[C@@H](OC2=O)[C@@H](C(=C)C)c2cc(C)c(o2)C1");
    assert!(a.iter().all(|x| !x.1));
}

#[test]
fn bad_smiles_is_an_error() {
    assert!(parse_smiles("C1CC").is_err());
    assert!(parse_smiles("C(C").is_err());
    assert!(parse_smiles("CC)").is_err());
    assert!(parse_smiles("[Xx]").is_err());
}

// ------------------------------------------------------------ rings

#[test]
fn ring_order_matches_rdkit() {
    assert_eq!(rings("c1ccccc1"), vec![vec![0, 5, 4, 3, 2, 1]]);
    assert_eq!(rings("c1ccc2ccccc2c1"), vec![vec![0, 9, 8, 3, 2, 1], vec![4, 5, 6, 7, 8, 3]]);
    assert_eq!(rings("C1CC2CCC1C2"), vec![vec![0, 1, 2, 6, 5], vec![3, 2, 6, 5, 4]]);
    assert_eq!(
        rings("C=C(C)[C@@H]1CCC2=C[C@@H](OC2=O)[C@@H](C(=C)C)c2cc(C)c(o2)C1"),
        vec![vec![3, 4, 5, 6, 7, 8, 12, 16, 21, 20, 22], vec![7, 6, 10, 9, 8], vec![17, 16, 21, 20, 18]]
    );
    assert!(rings("CCO").is_empty());
}

// ------------------------------------------------------------ SMARTS matching

#[test]
fn matching_follows_rdkit_semantics() {
    let cases = [
        ("CC", "C1CC1", true), // non-induced
        ("c/c", "c1ccccc1", true),
        ("C/C", "C=C", false),
        ("[C&H3&D1]", "CC", true),
        ("[C&H4]", "[H]C", true),
        ("[#7&a]", "c1ccncc1", true),
        ("[#7&a]", "CN", false),
        ("[O&-]", "C[N+](=O)[O-]", true),
        ("[N&+0]", "C[N+](=O)[O-]", false),
        ("[C@&H1](F)(Cl)Br", "F[C@@H](Cl)Br", true), // chirality ignored
        ("c:c", "c1ccccc1", true),
        ("c-c", "c1ccccc1-c1ccccc1", true),
        ("c-c", "c1ccccc1", false),
        ("C~O", "C=O", true),
        ("C#N", "CC#N", true),
        ("[C,N]=O", "CN=O", true),
        ("[!#6]", "CO", true),
        ("[#6&!a]", "c1ccccc1", false),
        ("O=C-[#7]", "CC(=O)N", true),
        ("c1ccccc1.O", "c1ccccc1O", true),
        ("[C&D4]", "CC(C)(C)C", true),
        ("[#8]-c1:c:[c&H1&D2&+0]:c:[c&H0&D2&+0]#[c&H0&D2&+0]:1", "COc1c#cccc1", true),
    ];
    for (q, s, want) in cases {
        assert_eq!(matches(q, s), want, "{q} in {s}");
    }
}

#[test]
fn unsupported_smarts_is_rejected() {
    assert!(parse_smarts("[$(CO)]").is_err());
    assert!(parse_smarts("C@C").is_err());
    assert!(parse_smarts("[13C]").is_err());
    assert!(parse_smarts("C1CC").is_err());
}

// ------------------------------------------------------------ screening

#[test]
fn fingerprint_never_rejects_a_match() {
    let queries = [
        "c1ccccc1",
        "[#8]-c1:c:[c&H1&D2&+0]:c:[c&H0&D2&+0]#[c&H0&D2&+0]:1",
        "O=C-[#7]",
        "[C&H3&D1]-[C&H0&D3&+0](=[O&H0&D1&+0])-O",
        "[#7&a]:c",
        "C~O",
        "[C,N]=O",
    ];
    let targets = ["COc1c#cccc1", "CC(=O)Nc1ccncc1", "CC(=O)OC", "c1ccccc1-c1ccccc1", "CN=O"];
    for s in targets {
        let t = Target::new(&parse_smiles(s).unwrap());
        let tfp = target_fp(&t);
        for smarts in queries {
            let q = parse_smarts(smarts).unwrap();
            if has_match(&q, &Plan::new(&q), &t) {
                let qfp = query_fp(&q);
                assert!(qfp.iter().zip(&tfp).all(|(q, t)| q & !t == 0), "{smarts} in {s}");
            }
        }
    }
}

#[test]
fn index_returns_exact_matches() {
    let smarts = ["c1ccccc1", "C=O", "[#7&a]", "C#N", "[O&-]"];
    let index = Index::build(smarts.iter().map(|s| parse_smarts(s).unwrap()).collect());
    let t = Target::new(&parse_smiles("O=Cc1ccncc1").unwrap());
    for limit in [1, 128] {
        let (_, found) = index.match_one(&t, &target_fp(&t), limit);
        assert_eq!(found, vec![1, 2], "posting limit {limit}");
    }
}
