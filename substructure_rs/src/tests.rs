//! Unit tests. Expected values were taken from RDKit 2026.03.6
//! (`Chem.MolFromSmiles`, `GetRingInfo().AtomRings()`, `HasSubstructMatch`).
//!
//! Author: Marcin Kowiel + Claude

use crate::fingerprint::{fp_kind, query_fp, target_fp, Fp, FP_KINDS};
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
    assert!(parse_smarts("[$()]").is_err() && parse_smarts("[$(C]").is_err() && parse_smarts("[$C]").is_err());
    assert!(parse_smarts("C->C").is_err() && parse_smarts("C<-C").is_err());
    assert!(parse_smarts("C1CC").is_err());
}

// ------------------------------------------------------------ phase 1 primitives

#[test]
fn ring_info_is_kept_on_atoms_and_bonds() {
    // Spiro centre: a 5- and a 4-ring; the bridgeheads of norbornane: two 5-rings.
    let t = parse_smiles("C1CCC2(C1)CCC2").unwrap().target_atoms();
    assert_eq!((t[3].rings, t[3].ring_size, t[3].ring_bonds), (2, 4, 4));
    assert_eq!((t[0].rings, t[0].ring_size, t[0].ring_bonds), (1, 5, 2));
    let m = parse_smiles("CC1CC1O").unwrap();
    assert_eq!(m.bonds.iter().map(|b| b.in_ring).collect::<Vec<_>>(), vec![false, true, true, false, true]); // ring closure bond last
    let t = m.target_atoms();
    assert_eq!((t[0].rings, t[0].ring_size, t[0].ring_bonds), (0, 0, 0));
    // Ring info survives the removal of explicit hydrogens (atom and bond indices shift).
    let m = parse_smiles("[H]C1CC1[H]").unwrap();
    assert_eq!(m.bonds.len(), 3);
    assert!(m.bonds.iter().all(|b| b.in_ring));
    assert!(m.target_atoms().iter().all(|a| a.rings == 1 && a.ring_size == 3 && a.ring_bonds == 2));
    // Cubane: six symmetrized SSSR rings, three at every atom.
    assert!(parse_smiles("C12C3C4C1C5C2C3C45").unwrap().target_atoms().iter().all(|a| a.rings == 3 && a.ring_size == 4));
}

#[test]
fn counted_properties_follow_rdkit() {
    // CH3-NH2 with explicit hydrogens: X counts them, `h` ignores [H] neighbours, `H` includes them.
    let t = parse_smiles("[H]C([H])([H])[NH2]").unwrap().target_atoms();
    assert_eq!((t[0].connections, t[0].implicit_h, t[0].total_h, t[0].degree, t[0].valence), (4, 3, 3, 1, 4));
    assert_eq!((t[1].connections, t[1].implicit_h, t[1].valence), (3, 2, 3));
    // A kept [H] atom (not removable) stays a neighbour: it is in degree, in H, not in h.
    let t = parse_smiles("[H][H]").unwrap().target_atoms();
    assert_eq!((t[0].degree, t[0].total_h, t[0].implicit_h, t[0].connections), (1, 1, 0, 1));
    // z and Z: heteroatom neighbours, aliphatic ones.
    let t = parse_smiles("Cc1ccncc1").unwrap().target_atoms();
    assert_eq!((t[3].hetero_nbrs, t[3].aliphatic_hetero_nbrs), (1, 0)); // next to the aromatic N
    assert_eq!((t[0].hetero_nbrs, t[0].aliphatic_hetero_nbrs), (0, 0));
    assert_eq!((t[4].hetero_nbrs, t[4].aliphatic_hetero_nbrs), (0, 0));
    let t = parse_smiles("CC(=O)N").unwrap().target_atoms();
    assert_eq!((t[1].hetero_nbrs, t[1].aliphatic_hetero_nbrs), (2, 2));
    // Aromatic valence is integral.
    assert_eq!(parse_smiles("c1ccccc1").unwrap().target_atoms()[0].valence, 4);
}

#[test]
fn rdkit_agrees_on_new_primitives() {
    let cases = [
        ("[R]", "C1CC1C", true),
        ("[R]", "CCC", false),
        ("[R0]", "CCC", true),
        ("[R2]", "c1ccc2ccccc2c1", true),
        ("[R3]", "C12C3C4C1C5C2C3C45", true),
        ("[R2]", "C1CC2CCC1C2", true),
        ("[r5]", "C1CCC2(C1)CCC2", true),
        ("[r4]", "C1CCC2(C1)CCC2", true),
        ("[r3]", "C1CCC2(C1)CCC2", false),
        ("[r]", "CCO", false),
        ("[x2]", "c1ccccc1C", true),
        ("[x3]", "c1ccc2ccccc2c1", true),
        ("[x]", "CC", false),
        ("[C;x3;R2]", "C1CC2CCC1C2", true),
        ("[X]", "[2H]C", true),
        ("[X4]", "C", true),
        ("[X3]", "[H]C([H])([H])[NH2]", true),
        ("[X3]", "c1ccccc1", true),
        ("[X4]", "[NH4+]", true),
        ("[v]", "[O-]C", true),
        ("[v4]", "C", true),
        ("[v4]", "c1ccccc1", true),
        ("[v5]", "CS(=O)(=O)C", false),
        ("[v6]", "CS(=O)(=O)C", true),
        ("[v3]", "c1cc[nH]c1", true),
        ("[h]", "[H]C([H])([H])[H]", true),
        ("[h]", "C", true),
        ("[h2]", "[NH2]C", true),
        ("[h]", "[2H]C#N", false),
        ("[H]", "[2H]C#N", true),
        ("[h0]", "O=C=O", true),
        ("[h]", "CC#N", true),
        ("[z]", "CC", false),
        ("[z]", "CO", true),
        ("[z2]", "COC", false),
        ("[z1]", "COC", true),
        ("[z]", "*C", true),
        ("[z2]", "CC(=O)N", true),
        ("[Z]", "c1ccncc1", false),
        ("[Z]", "Cc1ccncc1", false),
        ("[z]", "Cc1ccncc1", true),
        ("[Z]", "CS", true),
        ("[Z1]", "CC(=O)N", false),
        ("[z0]", "CC", true),
        ("[Z0]", "c1cnccc1", true),
        ("[13C]", "[13CH4]", true),
        ("[13C]", "C", false),
        ("[C]", "[13CH4]", true),
        ("[0C]", "[13CH4]", false),
        ("[0C]", "C", true),
        ("[13c]", "[13c]1ccccc1", true),
        ("[13c]", "c1ccccc1", false),
        ("[13#6]", "[13CH4]", true),
        ("[13]", "[13CH4]", true),
        ("[13]", "[2H]C", false),
        ("[2H]", "[2H]C", true),
        ("[2H]", "C", false),
        ("[H]", "[2H]C", true),
        ("[2H+]", "[2H+]", true),
        ("[!13C]", "[13CH4]", false),
        ("[!13C]", "CC", true),
        ("[C13]", "[13CH4]", true),
        ("[0]", "C", true),
        ("[0]", "[13CH4]", false),
        ("[14C]", "[13CH4]", false),
        ("[Dy]", "[Dy+3]", true),
        ("[D]", "[Dy+3]", false),
        ("[Db]", "[Db]", true),
        ("[Ds]", "[Ds]", true),
        ("[Rh]", "[Rh]", true),
        ("[R]", "[Rh]", false),
        ("[Re]", "[Re]", true),
        ("[Xe]", "[Xe]", true),
        ("[Xe]", "C", false),
        ("[Zn]", "[Zn+2]", true),
        ("[Zr]", "[Zn+2]", false),
        ("[Hg]", "C[Hg]Cl", true),
        ("[Hs]", "[Hs]", true),
        ("[Hf]", "[Hf]", true),
        ("[Ho]", "[Ho+3]", true),
        ("[Cn]", "[Cn]", true),
        ("[Se]", "c1cc[se]c1", true),
        ("[se]", "c1cc[se]c1", true),
        ("[Nh]", "N", true),
        ("[Nh]", "[NH3]", true),
        ("[Nh]", "[N]", false),
        ("C$C", "C$C", true),
        ("C~C", "C$C", true),
        ("CC", "C$C", false),
        ("C=C", "C$C", false),
        ("C#C", "C$C", false),
        ("A$A", "C$C", true),
        ("A", "C", true),
        ("a", "C", false),
        ("a", "c1ccccc1", true),
        ("AA", "CC", true),
        ("aa", "c1ccccc1", true),
        ("AA", "c1ccccc1", false),
        ("Aa", "Cc1ccccc1", true),
        ("Na", "[Na+]", false),
        ("Cc", "Cc1ccccc1", true),
        ("[#119]", "C", false),
        ("[#118]", "[Og]", true),
        ("[#300]", "C", false),
        ("[C:1]", "C", true),
        ("[C:1]-[N:2]", "CN", true),
        ("[H;R]", "C1CC1", false),
        ("[Hx]", "C1CC1", false),
        ("[H,N]", "C", false),
        ("[H,N]", "N", true),
        ("[!H]", "C", true),
        ("[H+]", "[H+]", true),
    ];
    for (q, s, want) in cases {
        assert_eq!(matches(q, s), want, "{q} in {s}");
    }
}

#[test]
fn bare_property_letters_use_rdkit_defaults() {
    use crate::smarts::{Expr, Prim, Prop};
    let prim = |s: &str| match parse_smarts(s).unwrap().atoms.remove(0).expr {
        Expr::Prim(p) => p,
        e => panic!("{s}: {e:?}"),
    };
    for (s, prop, lo, hi) in [
        ("[X]", Prop::Connections, 1, 1),
        ("[v]", Prop::Valence, 1, 1),
        ("[h]", Prop::ImplicitH, 1, 255),
        ("[z]", Prop::HeteroNbrs, 1, 255),
        ("[Z]", Prop::AliphaticHeteroNbrs, 1, 255),
        ("[R]", Prop::Rings, 1, 255),
        ("[r]", Prop::RingSize, 1, 255),
        ("[x]", Prop::RingBonds, 1, 255),
        ("[R0]", Prop::Rings, 0, 0),
        ("[X300]", Prop::Connections, 255, 255),
    ] {
        let Prim::Num { prop: p, lo: l, hi: h } = prim(s) else { panic!("{s}") };
        assert_eq!((p, l, h), (prop, lo, hi), "{s}");
    }
    assert!(matches!(prim("[D]"), Prim::Num { prop: Prop::Degree, lo: 1, hi: 1 }));
    assert!(matches!(prim("[Dy]"), Prim::Element { z: 66, .. }));
}

#[test]
fn element_symbols_beat_primitive_letters() {
    use crate::periodic::symbol_to_z;
    use crate::smarts::{Expr, Prim};
    for sym in crate::periodic::TABLE.iter().skip(1).map(|e| e.symbol) {
        let q = parse_smarts(&format!("[{sym}]"));
        // RDKit's lexer lacks these four: [Nh] is N&h, the others do not parse.
        if ["Nh", "Mc", "Ts", "Og"].contains(&sym) {
            assert_eq!(q.is_ok(), sym == "Nh", "{sym}");
            continue;
        }
        let q = q.unwrap_or_else(|e| panic!("{sym}: {e}"));
        let Expr::Prim(Prim::Element { z, .. }) = q.atoms[0].expr else { panic!("{sym}: {:?}", q.atoms[0].expr) };
        assert_eq!(Some(z), symbol_to_z(sym), "{sym}");
    }
    // Se, Si, Te, As carry no aliphatic restriction, so they match their aromatic forms.
    assert!(matches("[Se]", "c1cc[se]c1"));
    assert!(!matches("[Cl]", "CC"));
}

#[test]
fn isotope_atomic_number_and_other_small_parser_fixes() {
    assert!(parse_smarts("[13C]").is_ok());
    assert!(parse_smarts("[#118]").is_ok() && parse_smarts("[#119]").is_ok());
    assert!(parse_smarts("[#]").is_err() && parse_smarts("[#-1]").is_err());
    assert!(parse_smarts("[#4294967296]").is_err());
    assert!(parse_smarts("[Mc]").is_err() && parse_smarts("[Q]").is_err() && parse_smarts("[xe]").is_err());
    assert!(parse_smarts("C1CC").is_err());
    // `#263` must not wrap around to nitrogen.
    assert!(!matches("[#263]", "N"));
    // Unbracketed A and a; `$` is a quadruple bond; `~` matches it, the default bond does not.
    let q = parse_smarts("A$A").unwrap();
    assert_eq!(q.bonds[0].query, crate::smarts::BondQuery::QUADRUPLE);
    assert_eq!(parse_smarts("Aa").unwrap().atoms.len(), 2);
    assert_eq!(bonds("C$C"), vec![BondType::Quadruple]);
}

#[test]
fn compiled_new_primitives_agree_with_expressions() {
    use crate::smarts::compile_atom;
    let smarts = [
        "[R]", "[!R]", "[R0]", "[R2]", "[!R0]", "[!R2]", "[r5]", "[r]", "[x2]", "[x]", "[!x]", "[X3]", "[X]", "[v4]", "[h]", "[!h]", "[h1]",
        "[z]", "[!z]", "[z2]", "[Z]", "[Z1]", "[!Z0]", "[13C]", "[0C]", "[13]", "[C;R]", "[C;!R]", "[N;X3;h0]", "[c;r6;x2]", "[R,X4]",
        "[#6;R1]", "[X1,X2]", "[!X3]", "[R300]", "[!R300]",
    ];
    let mols = [
        "CC(=O)Nc1ccncc1", "C[N+](=O)[O-]", "C12C3C4C1C5C2C3C45", "C1CCC2(C1)CCC2", "[13CH4].[2H]C", "c1cc[nH]c1", "[H]C([H])([H])[NH2]", "OP(=O)(O)O",
        "c1ccc2ccccc2c1", "[Zn+2]",
    ];
    let mut compiled = 0;
    for s in smarts {
        let q = parse_smarts(s).unwrap();
        for qa in &q.atoms {
            let Some(terms) = compile_atom(&qa.expr) else { continue };
            compiled += 1;
            for m in mols {
                for t in parse_smiles(m).unwrap().target_atoms() {
                    let p = t.packed();
                    let fast = terms.iter().any(|&(mask, value)| p & mask == value);
                    assert_eq!(fast, qa.expr.matches(&t), "{s} on an atom of {m}");
                }
            }
        }
    }
    assert!(compiled >= 30, "only {compiled} atoms compiled");
}

#[test]
fn fingerprint_never_rejects_a_new_primitive_match() {
    let queries = [
        "[C;R]", "[C;X4;h3]", "[c;r6;x2]", "[N;z1]", "[C;Z1]", "[#6;v4]", "[13C]", "[0C]", "[Se]", "[2H]", "A$A", "[C;R2]-[C;R2]", "[R]~[R]", "[Dy]",
        "[#6;X3]-[#8;h1]", "C1CC1", "[C;R1]1[C;R1][C;R1]1",
    ];
    let targets = [
        "C1CC1C", "c1ccc2ccccc2c1", "CC(=O)O", "C[13CH3]", "[2H]C", "c1cc[se]c1", "C$C", "[Dy+3]", "C12C3C4C1C5C2C3C45", "CC(C)(C)N", "OC(=O)C1CC1",
    ];
    for kind in &FP_KINDS {
        for s in targets {
            let t = Target::new(&parse_smiles(s).unwrap());
            let tfp: Fp<16> = target_fp(kind, &t);
            for smarts in queries {
                let q = parse_smarts(smarts).unwrap();
                if has_match(&q, &Plan::new(&q), &t) {
                    let qfp: Fp<16> = query_fp(kind, &q);
                    assert!(qfp.iter().zip(&tfp).all(|(q, t)| q & !t == 0), "{} {smarts} in {s}", kind.name);
                }
            }
        }
    }
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
    for kind in &FP_KINDS {
        for s in targets {
            let t = Target::new(&parse_smiles(s).unwrap());
            let tfp: Fp<16> = target_fp(kind, &t);
            for smarts in queries {
                let q = parse_smarts(smarts).unwrap();
                if has_match(&q, &Plan::new(&q), &t) {
                    let qfp: Fp<16> = query_fp(kind, &q);
                    assert!(qfp.iter().zip(&tfp).all(|(q, t)| q & !t == 0), "{} {smarts} in {s}", kind.name);
                }
            }
        }
    }
}

#[test]
fn cycles_and_branches_screen_out_non_matches() {
    // Six-membered ring against cyclopentane plus a chain; a tertiary centre against a chain.
    for (kind, smarts, smiles) in
        [("paths4+cycles", "C1CCCCC1", "C1CCCC1CCCCCC"), ("paths4+branches", "CC(C)C", "CCCCCC")]
    {
        let kind = fp_kind(kind).unwrap();
        let tfp: Fp<64> = target_fp(&kind, &Target::new(&parse_smiles(smiles).unwrap()));
        let qfp: Fp<64> = query_fp(&kind, &parse_smarts(smarts).unwrap());
        assert!(qfp.iter().zip(&tfp).any(|(q, t)| q & !t != 0), "{} {smarts} in {smiles}", kind.name);
    }
}

#[test]
fn index_returns_exact_matches() {
    let smarts = ["c1ccccc1", "C=O", "[#7&a]", "C#N", "[O&-]"];
    let kind = fp_kind("paths4").unwrap();
    let index = Index::<32>::build(&kind, crate::matcher::Order::RareRoot, smarts.iter().map(|s| parse_smarts(s).unwrap()).collect());
    let t = Target::new(&parse_smiles("O=Cc1ccncc1").unwrap());
    for limit in [1, 128] {
        let found = index.match_one(&t, &target_fp(&kind, &t), limit).found;
        assert_eq!(found, vec![1, 2], "posting limit {limit}");
    }
}

#[test]
fn compiled_atom_tests_agree_with_expressions() {
    use crate::smarts::compile_atom;
    let smarts = [
        "[C&H3&D1]", "[c&H0&D2&+0]", "[#7&a]", "[#8&-]", "[N&+]", "[C,N]", "[!#6]", "[#6&!a]", "[!a]",
        "[C,N;H1]", "[#6&H1,#7&H0]", "[C&H2&H3]", "*", "[Cl]", "[O&H1&D1&+0]",
    ];
    let mols = ["CC(=O)Nc1ccncc1", "C[N+](=O)[O-]", "OCl(=O)(=O)=O", "c1cc[nH]c1", "[NH4+].[Cl-]"];
    let mut compiled = 0;
    for s in smarts {
        let q = parse_smarts(s).unwrap();
        for qa in &q.atoms {
            let Some(terms) = compile_atom(&qa.expr) else { continue };
            compiled += 1;
            for m in mols {
                for t in parse_smiles(m).unwrap().target_atoms() {
                    let p = t.packed();
                    let fast = terms.iter().any(|&(mask, value)| p & mask == value);
                    assert_eq!(fast, qa.expr.matches(&t), "{s} on an atom of {m}");
                }
            }
        }
    }
    assert!(compiled >= 12, "only {compiled} atoms compiled");
}

#[test]
fn every_atom_order_finds_the_same_matches() {
    use crate::matcher::{LabelFreq, ORDERS};
    let smarts = [
        "c1ccccc1", "[C&H3&D1]-[C&H0&D3&+0](=[O&H0&D1&+0])-O", "[#7&a]:c", "C~O", "[C,N]=O", "O=C-[#7]",
        "[#8]-c1:c:[c&H1&D2&+0]:c:[c&H0&D2&+0]#[c&H0&D2&+0]:1", "c1ccccc1.O", "[!#6]", "CC(C)C",
    ];
    let queries: Vec<_> = smarts.iter().map(|s| parse_smarts(s).unwrap()).collect();
    let freq = LabelFreq::new(&queries);
    let targets = ["COc1c#cccc1", "CC(=O)Nc1ccncc1", "CC(=O)OC", "c1ccccc1-c1ccccc1", "CN=O", "c1ccccc1O", "CC(C)CC"];
    for s in targets {
        let t = Target::new(&parse_smiles(s).unwrap());
        for q in &queries {
            let want = has_match(q, &Plan::new(q), &t);
            for &(name, order) in &ORDERS {
                assert_eq!(has_match(q, &Plan::with_order(q, order, &freq), &t), want, "{name} {s}");
            }
        }
    }
}

#[test]
fn fingerprints_do_not_depend_on_atom_order() {
    // The same molecules written from different starting atoms.
    let pairs = [
        ("OCC(=O)N", "NC(=O)CO"),
        ("c1ccncc1C(=O)O", "OC(=O)c1cnccc1"),
        ("C1CC2CCC1C2", "C1CC2CC1CC2"),
        ("CC(C)(C)c1ccc(O)cc1", "Oc1ccc(cc1)C(C)(C)C"),
    ];
    for kind in &FP_KINDS {
        for (a, b) in pairs {
            let fa: Fp<64> = target_fp(kind, &Target::new(&parse_smiles(a).unwrap()));
            let fb: Fp<64> = target_fp(kind, &Target::new(&parse_smiles(b).unwrap()));
            assert_eq!(fa, fb, "{} {a} vs {b}", kind.name);
        }
    }
}

// ------------------------------------------------------------ index on disk

#[test]
fn saved_index_loads_and_matches_the_same() {
    use crate::store::{Queries, StoredIndex};
    use crate::Screen;

    let smarts = ["[CH3]", "c1ccccc1", "C(=O)[OH]", "[N+](=O)[O-]", "[Cl]", "C#N"];
    let smiles = ["Cc1ccccc1", "CC(=O)O", "O=[N+]([O-])c1ccccc1Cl", "CCCC", "N#CC"];
    let queries = Queries::parse(smarts.iter().map(|s| s.to_string()).collect()).unwrap();
    let built = StoredIndex::build(queries, "paths4+branches", 2048, "rare-root").unwrap();
    let path = std::env::temp_dir().join(format!("substructure_rs_test_{}.idx", std::process::id()));
    built.save(&path).unwrap();
    let loaded = StoredIndex::load(&path);
    std::fs::remove_file(&path).unwrap();
    let loaded = loaded.unwrap();

    assert_eq!((loaded.queries(), loaded.fp_kind(), loaded.fp_bits()), (6, "paths4+branches", 2048));
    assert_eq!(loaded.nbytes(), built.nbytes());
    let targets = Screen::targets(&smiles.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap();
    let found = |index: &StoredIndex| -> Vec<Vec<u32>> {
        index.match_many(&targets, 128).unwrap().into_iter().map(|f| f.found).collect()
    };
    assert_eq!(found(&loaded), found(&built));
    assert_eq!(found(&loaded), vec![vec![0, 1], vec![0, 2], vec![1, 3, 4], vec![0], vec![0, 5]]);
}

#[test]
fn truncated_index_is_an_error() {
    use crate::store::{Queries, StoredIndex};

    let queries = Queries::parse(vec!["CC".into(), "c1ccccc1".into()]).unwrap();
    let index = StoredIndex::build(queries, "paths4", 512, "rare-root").unwrap();
    let path = std::env::temp_dir().join(format!("substructure_rs_trunc_{}.idx", std::process::id()));
    index.save(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
    let loaded = StoredIndex::load(&path);
    std::fs::remove_file(&path).unwrap();
    assert!(loaded.is_err());
}

// ------------------------------------------------------------ bond logic and ranges

#[test]
fn bond_expressions_match_rdkit() {
    for (smarts, smiles, want) in [
        ("C-,=C", "CC", true),
        ("C-,=C", "C=C", true),
        ("C-,=C", "C#C", false),
        ("C-;!@C", "CC", true),
        ("C-;!@C", "C1CC1", false),
        ("C-;!@C", "CC1CC1", true),
        ("C!-C", "CC", false),
        ("C!-C", "C=C", true),
        ("C=&@C", "C1=CCC1", true),
        ("C=&@C", "C=C", false),
        ("C~;!@C", "C1CC1", false),
        ("C~;!@C", "CC", true),
        ("C-!@C", "CC1CC1", true),
        ("C!:C", "c1ccccc1", false),
        ("C!:C", "C=C", true),
        ("C@C", "C1CC1", true),
        ("C@C", "CC", false),
        ("C!@C", "C1CCC1C", true),
        ("c@c", "c1ccccc1", true),
        ("c!@c", "c1ccccc1", false),
        ("c!@c", "c1ccccc1c1ccccc1", true),
        ("c!@c", "c1ccccc1C", false),
        ("[#6]!@[#6]", "C1CC1C", true),
        ("C/C", "CC", true),
        ("C/C", "c1ccccc1", false),
        ("C\\C", "C=C", false),
        ("C!/C", "C=C", true),
        ("c/c", "c1ccccc1", true),
        ("C-1CCCCC=1", "C1CCCCC1", true),
        ("C=1CCCCC-1", "C1CCCCC1", false),
        ("C=1CCCCC-1", "C1=CCCCC1", true),
        ("C-,=1CCCCC1", "C1=CCCCC1", true),
        ("C1CCCCC=,-1", "C1CCCCC1", true),
        ("C1CCCC!@1", "C1CCCCC1", false),
        ("C1CCCCC!@1", "C1CCCCC1", false),
        ("C1-,=C-C-C-C1", "C1=CCCC1", true),
        ("C1-,=CCCC1", "C1=CCCC1", true),
        ("C1(-,=C)CCCCC1", "CC1CCCCC1", true),
        ("C(!@C)C", "C1CCCCC1C", true),
        ("C!@C", "C1CCC1", false),
        ("C1CC1!@C", "CC1CC1", true),
        ("C-1CC-1", "C1CC1", true),
        ("C-&=C", "C=C", false),
        ("C!-&!=C", "C#C", true),
        ("C!!-C", "CC", true),
        ("[C]-,=[C]", "CC", true),
        ("C=,#C", "C#C", true),
        ("C=,#;@C", "C#C", false),
        ("C=,#;!@C", "C#C", true),
        ("C-@=C", "C=C", false),
        ("C-;@C", "C1CC1", true),
        ("C@;!@C", "C1CC1", false),
        ("C-,@C", "CC", true),
        ("*!@*", "C1CC1", false),
        ("C!@*", "C1CC1C", true),
        ("C!@*", "C1CC1", false),
        ("c1ccccc1!@C", "Cc1ccccc1", true),
        ("C~C", "[Mo]$[Mo]", false),
        ("[Mo]~[Mo]", "[Mo]$[Mo]", true),
        ("[Mo]!-[Mo]", "[Mo]$[Mo]", true),
        ("[Mo]!$[Mo]", "[Mo]$[Mo]", false),
        ("[Mo]-,$[Mo]", "[Mo]$[Mo]", true),
        ("[Cu]!-N", "[Cu]<-N", true),
        ("[Cu]~N", "N->[Cu]", true),
    ] {
        assert_eq!(matches(smarts, smiles), want, "{smarts} on {smiles}");
    }
}

#[test]
fn ranges_match_rdkit() {
    let mols = ["CC(=O)Nc1ccncc1", "C12C3C4C1C5C2C3C45", "C1CCC2(C1)CCC2", "[13CH4].[2H]C", "c1cc[nH]c1", "OP(=O)(O)O", "c1ccc2ccccc2c1", "[O-3]", "[O-2]", "[O-]", "[Mg]", "[Na+]", "[Mg+2]", "[Al+3]", "C[N+](=O)[O-]", "CC(C)(C)C"];
    for (smarts, want) in [
        ("[D{1-2}]", "1011111000000011"),
        ("[D{-1}]", "1001010111111111"),
        ("[D{-2}]", "1011111111111111"),
        ("[D{2-}]", "1110111000000011"),
        ("[D{1-}]", "1111111000000011"),
        ("[D{0-0}]", "0001000111111100"),
        ("[D{2-1}]", "0000000000000000"),
        ("[D{0-}]", "1111111111111111"),
        ("[D{3-300}]", "1110011000000011"),
        ("[D{300-400}]", "0000000000000000"),
        ("[D{-0}]", "0001000111111100"),
        ("[D{0-300}]", "1111111111111111"),
        ("[h{1-2}]", "1110111000000000"),
        ("[h{-1}]", "1111111111111111"),
        ("[h{-2}]", "1111111111111111"),
        ("[h{2-}]", "1011000000000011"),
        ("[h{1-}]", "1111111000000011"),
        ("[h{0-0}]", "1011011111111111"),
        ("[h{2-1}]", "0000000000000000"),
        ("[h{0-}]", "1111111111111111"),
        ("[h{3-300}]", "1001000000000011"),
        ("[h{300-400}]", "0000000000000000"),
        ("[h{-0}]", "1011011111111111"),
        ("[h{0-300}]", "1111111111111111"),
        ("[R{1-2}]", "1010101000000000"),
        ("[R{-1}]", "1011111111111111"),
        ("[R{-2}]", "1011111111111111"),
        ("[R{2-}]", "0110001000000000"),
        ("[R{1-}]", "1110101000000000"),
        ("[R{0-0}]", "1001010111111111"),
        ("[R{2-1}]", "0000000000000000"),
        ("[R{0-}]", "1111111111111111"),
        ("[R{3-300}]", "0100000000000000"),
        ("[R{300-400}]", "0000000000000000"),
        ("[R{-0}]", "1001010111111111"),
        ("[R{0-300}]", "1111111111111111"),
        ("[r{1-2}]", "0000000000000000"),
        ("[r{-1}]", "1001010111111111"),
        ("[r{-2}]", "1001010111111111"),
        ("[r{2-}]", "1110101000000000"),
        ("[r{1-}]", "1110101000000000"),
        ("[r{0-0}]", "1001010111111111"),
        ("[r{2-1}]", "0000000000000000"),
        ("[r{0-}]", "1111111111111111"),
        ("[r{3-300}]", "1110101000000000"),
        ("[r{300-400}]", "0000000000000000"),
        ("[r{-0}]", "1001010111111111"),
        ("[r{0-300}]", "1111111111111111"),
        ("[v{1-2}]", "1001010000000010"),
        ("[v{-1}]", "0001000111111110"),
        ("[v{-2}]", "1001010111111110"),
        ("[v{2-}]", "1111111000000011"),
        ("[v{1-}]", "1111111000000011"),
        ("[v{0-0}]", "0000000111111100"),
        ("[v{2-1}]", "0000000000000000"),
        ("[v{0-}]", "1111111111111111"),
        ("[v{3-300}]", "1111111000000011"),
        ("[v{300-400}]", "0000000000000000"),
        ("[v{-0}]", "0000000111111100"),
        ("[v{0-300}]", "1111111111111111"),
        ("[X{1-2}]", "1001010000000010"),
        ("[X{-1}]", "1001010111111110"),
        ("[X{-2}]", "1001010111111110"),
        ("[X{2-}]", "1111111000000011"),
        ("[X{1-}]", "1111111000000011"),
        ("[X{0-0}]", "0000000111111100"),
        ("[X{2-1}]", "0000000000000000"),
        ("[X{0-}]", "1111111111111111"),
        ("[X{3-300}]", "1111111000000011"),
        ("[X{300-400}]", "0000000000000000"),
        ("[X{-0}]", "0000000111111100"),
        ("[X{0-300}]", "1111111111111111"),
        ("[x{1-2}]", "1010101000000000"),
        ("[x{-1}]", "1001010111111111"),
        ("[x{-2}]", "1011111111111111"),
        ("[x{2-}]", "1110101000000000"),
        ("[x{1-}]", "1110101000000000"),
        ("[x{0-0}]", "1001010111111111"),
        ("[x{2-1}]", "0000000000000000"),
        ("[x{0-}]", "1111111111111111"),
        ("[x{3-300}]", "0110001000000000"),
        ("[x{300-400}]", "0000000000000000"),
        ("[x{-0}]", "1001010111111111"),
        ("[x{0-300}]", "1111111111111111"),
        ("[z{1-2}]", "1000110000000010"),
        ("[z{-1}]", "1111111111111111"),
        ("[z{-2}]", "1111111111111111"),
        ("[z{2-}]", "1000010000000010"),
        ("[z{1-}]", "1000110000000010"),
        ("[z{0-0}]", "1111101111111101"),
        ("[z{2-1}]", "0000000000000000"),
        ("[z{0-}]", "1111111111111111"),
        ("[z{3-300}]", "0000010000000000"),
        ("[z{300-400}]", "0000000000000000"),
        ("[z{-0}]", "1111101111111101"),
        ("[z{0-300}]", "1111111111111111"),
        ("[Z{1-2}]", "1000010000000010"),
        ("[Z{-1}]", "1111111111111111"),
        ("[Z{-2}]", "1111111111111111"),
        ("[Z{2-}]", "1000010000000010"),
        ("[Z{1-}]", "1000010000000010"),
        ("[Z{0-0}]", "1111101111111101"),
        ("[Z{2-1}]", "0000000000000000"),
        ("[Z{0-}]", "1111111111111111"),
        ("[Z{3-300}]", "0000010000000000"),
        ("[Z{300-400}]", "0000000000000000"),
        ("[Z{-0}]", "1111101111111101"),
        ("[Z{0-300}]", "1111111111111111"),
        ("[+{1-2}]", "0000000000011010"),
        ("[-{1-2}]", "0000000011000010"),
        ("[+{-1}]", "1111111111110011"),
        ("[-{-1}]", "1111111001111111"),
        ("[+{-2}]", "1111111111111011"),
        ("[-{-2}]", "1111111011111111"),
        ("[+{2-}]", "0000000000001100"),
        ("[-{2-}]", "0000000110000000"),
        ("[+{0-0}]", "1111111000100011"),
        ("[-{0-0}]", "1111111000100011"),
        ("[+{2-1}]", "0000000000000000"),
        ("[-{2-1}]", "0000000000000000"),
        ("[+{0-}]", "1111111000111111"),
        ("[-{0-}]", "1111111111100011"),
        ("[+{1-300}]", "0000000000011110"),
        ("[-{1-300}]", "0000000111000010"),
        ("[+{200-300}]", "0000000000000000"),
        ("[-{200-300}]", "0000000000000000"),
        ("[C;D{2-3}]", "1110000000000000"),
        ("[!D{1-2}]", "1111011111111111"),
        ("[C,N;R{1-2}]", "0010000000000000"),
        ("[!+{1-2}]", "1111111111100111"),
        ("[D{1-2}&R]", "1010101000000000"),
        ("[CR{1-2}]", "0010000000000000"),
        ("[N+{0-1}H3]", "0000000000000000"),
    ] {
        let got: String = mols.iter().map(|m| if matches(smarts, m) { '1' } else { '0' }).collect();
        assert_eq!(got, want, "{smarts}");
    }
}

#[test]
fn bond_and_range_syntax_errors_follow_rdkit() {
    for s in ["C-&C", "C&-C", "C-,C", "C;-C", "C/?C", "C-(C)C", "C-.C", "C-", "[H{1-2}]", "[D{2}]", "[D{-}]", "[D{}]", "[D{01-2}]", "[D{1 -2}]", "[#6{1-2}]", "[++{1-2}]", "[+2{1-2}]", "[D1{1-2}]", "[D{1-2}{1-2}]", "[D{1-2", "[D{a-2}]", "[D{1-2147483640}]", "[D2147483640]", "[+2147483640]", "[CH{1-2}]", "[a{1-2}]", "C!", "C=!", "C1CCCCC=1-"] {
        assert!(parse_smarts(s).is_err(), "{s}");
    }
    for s in ["C-,=C", "C-;!@C", "C=&@C", "[D{1-2147483639}]", "[D2147483639]", "[D{0-0}]", "[+{-0}]", "[R{-0}]", "[r{-2}]", "[C+{0-1}]", "C1CCCC!@1", "C!!-C", "C-@=C"] {
        assert!(parse_smarts(s).is_ok(), "{s}");
    }
}

#[test]
fn bond_queries_reduce_to_kind_ring_sets() {
    use crate::smarts::{bond_code, BondQuery};
    let one = |s: &str| parse_smarts(s).unwrap().bonds[0].query;
    assert_eq!(one("C-C"), BondQuery::SINGLE);
    assert_eq!(one("CC"), BondQuery::SINGLE_OR_AROMATIC);
    assert_eq!(one("C/C"), BondQuery::SINGLE_OR_AROMATIC);
    assert_eq!(one("C-,:C"), BondQuery::SINGLE_OR_AROMATIC);
    assert_eq!(one("C~C"), BondQuery::ANY);
    assert_eq!(one("C!!-C"), BondQuery::SINGLE);
    assert_eq!(one("C@C"), BondQuery::RING);
    assert_eq!(one("C~;@C"), BondQuery::RING);
    let q = one("C-;!@C");
    assert!(q.matches(bond_code(BondType::Single, false)));
    assert!(!q.matches(bond_code(BondType::Single, true)));
    assert!(!q.matches(bond_code(BondType::Double, false)));
    // Opposite ring-closure bonds: the opening digit wins.
    assert_eq!(parse_smarts("C=1CCCCC-1").unwrap().bonds.last().unwrap().query, BondQuery::DOUBLE);
}

#[test]
fn complex_bonds_never_pin_a_fingerprint_label() {
    // Screening must keep every true match, for every fingerprint kind.
    for (sma, smi) in [
        ("C-,=C", "C=C"), ("C-,=C", "CC"), ("C!-C", "C#C"), ("C~C", "C=C"), ("C-;!@C", "CC"), ("C=,#C", "C#C"),
        ("C!:CC", "C=CC"), ("c!@c", "c1ccccc1c1ccccc1"), ("C-,=C-C", "CC=C"), ("C=@C", "C1=CCC1"), ("C-@C", "C1CC1"),
        ("C1-,=C-C-C-C1", "C1=CCCC1"), ("C!-,!=C", "C#C"), ("[#6]-,:[#6]", "c1ccccc1"),
    ] {
        assert!(matches(sma, smi), "{sma} {smi}");
        let t = Target::new(&parse_smiles(smi).unwrap());
        let q = parse_smarts(sma).unwrap();
        for kind in FP_KINDS.iter() {
            let tfp: Fp<16> = target_fp(kind, &t);
            let qfp: Fp<16> = query_fp(kind, &q);
            assert!(qfp.iter().zip(&tfp).all(|(q, t)| q & !t == 0), "{} screened out {sma} in {smi}", kind.name);
        }
    }
}

// ------------------------------------------------------------ phase 3: recursive SMARTS

#[test]
fn recursive_smarts_parses_like_rdkit() {
    for ok in [
        "[$(CO)]", "[$(C.C)]", "[$(C1CC1)]", "[$(*)]", "[$(C),N]", "[$(C);N]", "[C$(C)]", "[$(C)C]", "[$([C;$(C=O)]N)]", "[!$(C=O)]",
        "[$(CC)$(CC)]", "[$(C=O)&$(CO)]", "C[$(C)]N", "[13$(C)]",
    ] {
        assert!(parse_smarts(ok).is_ok(), "{ok}");
    }
    for bad in [
        "[$()]", "[$(C]", "[$C]", "[$(C)(C)]", "[!$!$(C)]", "[$ (C)]", "[$(C )]", "[$(C-)]", "[$(-C)]", "[$(C.)]", "[$(C(C)]", "[$(=C)]",
        "[$(C:1)]", "[$($(C))]", "[$(C1)]", "[$((C))]", "[$(C)", "$(C)",
    ] {
        assert!(parse_smarts(bad).is_err(), "{bad}");
    }
}

#[test]
fn recursive_smarts_matches() {
    let cases = [
        ("[$(CO)]", "CO", true),
        ("[$(CO)]", "CC", false),
        ("[$(C=O)]N", "CC(=O)N", true),
        ("[$(C=O)]N", "CCN", false),
        ("[C;$(C=O)]", "CC(=O)C", true),
        ("[C;$(C=O)]", "CCO", false),
        ("[C;!$(C=O)]", "CCO", true),
        ("[C;!$(C=O)]", "C=O", false),
        ("[!$(C=O)]", "O=C", true),
        ("[$(C=O),$(C#N)]", "CC#N", true),
        ("[$(C=O),$(C#N)]", "CC", false),
        ("[$([C;$(C=O)]N)]", "CC(=O)N", true),
        ("[$([C;$(C=O)]N)]", "CCN", false),
        // the inner query may reuse the outer match's atoms
        ("[$(CO)]O", "CO", true),
        ("[$(C=O)]=O", "C=O", true),
        // a query atom pinned by a recursion is distinct from the others
        ("[$(C)][$(C)]", "C", false),
        ("[$(C)][$(C)]", "CC", true),
        // first inner atom is the root, wherever it sits in the inner chain
        ("[$(C(=O)N)]", "NC=O", true),
        ("[$(N=C)]", "C=N", true),
        ("[$(NC=O)]", "O=CN", true),
        ("[$(NC=O)]", "CN", false),
        ("[$(C1CC1)]", "C1CC1", true),
        ("[$(C1CC1)]", "CCC", false),
        // disconnected inner query
        ("[$(C.C)]", "CC", true),
        ("[$(C.C)]", "C", false),
        ("[$(C.O)]", "C.O", true),
        ("[$(C.O)]", "C", false),
        ("[$([NX3;H2,H1;!$(NC=O)])]", "CCN", true),
        ("[$([NX3;H2,H1;!$(NC=O)])]", "CC(=O)N", false),
        ("[CX3](=O)[OX2H1]", "CC(=O)O", true),
        ("[$([OH]-c),$([OH]-C=O)]", "CC(=O)O", true),
        ("[$([OH]-c),$([OH]-C=O)]", "CCO", false),
        ("[$([OH]-c),$([OH]-C=O)]", "Oc1ccccc1", true),
        // deep nesting
        ("[$([C;$(C-[N;!$(N-C=O)])]),$(O)]", "CCN", true),
        ("[$([C;$(C-[N;!$(N-C=O)])]),$(O)]", "CC(=O)NC", true),
    ];
    for (q, s, want) in cases {
        assert_eq!(matches(q, s), want, "{q} in {s}");
    }
}

#[test]
fn recursive_known_facts_are_implied() {
    let q = parse_smarts("[$(c)]").unwrap();
    assert_eq!(q.atoms[0].known.z, Some(6));
    assert_eq!(q.atoms[0].known.aromatic, Some(true));
    let q = parse_smarts("[!$(c)]").unwrap();
    assert_eq!(q.atoms[0].known.z, None);
    let q = parse_smarts("[$(c),$(C)]").unwrap();
    assert_eq!(q.atoms[0].known.z, None);
    assert_eq!(parse_smarts("[$([C;$(C=O)]N);$(CC)]C[$(O)]").unwrap().n_rec, 4);
}

#[test]
fn fingerprint_never_rejects_a_recursive_match() {
    let queries = [
        "[$(C(=O)N)]", "[C;$(C=O);$(CN)]", "[$([C;$(C(=O)[OH])]C)]", "[$(C=O),$(C#N)]", "[!$(C=O)]C", "[$(c1ccccc1)]", "[$(C1CC1)]C", "[$(CC)][$(CC)]",
        "[$(C(C)(C)C)]", "[$(C.C)]", "[$([OH]-c),$([OH]-C=O)]", "[$([NX3;H2,H1;!$(NC=O)])]", "[$(CCC)]-[$(CCC)]",
    ];
    let smiles = ["CC(=O)N", "CC(=O)O", "CC#N", "c1ccccc1O", "CC(C)(C)C", "C1CC1C", "CCCCCC", "CNC=O", "OC(=O)CC(=O)O", "CCN", "C"];
    for kind in FP_KINDS {
        for s in queries {
            let q = parse_smarts(s).unwrap();
            let plan = Plan::new(&q);
            let qfp: Fp<64> = query_fp(&kind, &q);
            for m in smiles {
                let t = Target::new(&parse_smiles(m).unwrap());
                if has_match(&q, &plan, &t) {
                    let tfp: Fp<64> = target_fp(&kind, &t);
                    assert!(qfp.iter().zip(&tfp).all(|(q, t)| q & !t == 0), "{} {s} in {m}", kind.name);
                }
            }
        }
    }
}
