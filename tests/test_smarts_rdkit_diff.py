"""substructure_rs agrees with RDKit's HasSubstructMatch on the atom primitives
added in phase 1, and (phase 2) on bond expressions (`! & , ;`, `@`, `/`, `\\`) and
range queries (`D{1-2}`, `R{-1}`, `+{2-}`...). Phase 1 covered ring (R r x),
valence/connectivity (X v h z Z), isotopes, two-letter element symbols versus primitive letters, `$`, `A`, `a`, `[#n]`.

Every (SMARTS, SMILES) pair goes through the Rust posting-list index
(fingerprint screen, then exact match), so a fingerprint false negative shows
up as a missing pair as well.
"""

import itertools
import re
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import substructure_rs
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ELEMENTS = [Chem.GetPeriodicTable().GetElementSymbol(z) for z in range(1, 119)]

TARGETS = [
    # rings
    "c1ccccc1", "c1ccc2ccccc2c1", "c1ccc2cc3ccccc3cc2c1", "c1ccc2c(c1)ccc1ccccc12",
    "C12C3C4C1C5C2C3C45", "C1CCC2(C1)CCC2", "C1CC2CCC1C2", "C1C2CC3CC1CC(C2)C3",
    "C1CC1", "C1CO1", "C1CCCCC1", "C1CCCCCCCCCCC1", "C1CC2CCC1CC2", "c1ccc2[nH]ccc2c1",
    "c1ccncc1", "c1cc[nH]c1", "c1ccoc1", "c1ccsc1", "c1ncc2nc[nH]c2n1",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(=O)Oc1ccccc1C(=O)O", "C1=CC=C2C=CC=C2C=C1",
    "C1CCC2C(C1)CCC1C2CCC2CCCC12", "c1cc2cccc3c2c(c1)CC3", "C1=CCC=CC1",
    "c1ccc(cc1)-c1ccccc1", "C1CC2(C1)CC1(C2)C1CC1", "C12CC3CC(CC(C3)C1)C2",
    "c1cc[se]c1", "c1cc[te]c1", "c1cc[as]c1", "c1cc[siH]c1", "C1CCNCC1", "O=C1CCCN1",
    "O=c1cc[nH]cc1", "C1=CNC=C1", "[nH]1cccc1",
    # acyclic, hetero
    "CC(=O)Nc1ccncc1[N+](=O)[O-]", "OCC(N)c1ccsc1Cl", "CC(C)(C)C", "CCO", "CC#N", "C=C=C",
    "CS(=O)(=O)O", "CS(C)=O", "OP(=O)(O)O", "P(=O)(O)(O)c1ccccc1", "CN(C)C=O", "NC(N)=O",
    "C[N+](C)(C)C", "[O-]C(=O)C", "C[N+](=O)[O-]", "C[n+]1ccccc1", "[O-][n+]1ccccc1",
    "OCl(=O)(=O)=O", "FC(F)(F)Cl", "BrCCI", "OB(O)c1ccccc1", "C[SiH2]C", "[SiH4]", "C[Se]C",
    "[NH4+]", "[OH-]", "[OH3+]", "CC[O-]", "C=[N+]=[N-]", "CN=[N+]=[N-]", "S=C=S", "O=S(=O)(O)O",
    "C1CCCCC1C", "CC1CC1", "C1=CCCC1", "C=CC=C", "C/C=C/C", "C/C=C\\C", "CC1CCC1C(=O)N", "c1ccccc1-c1ccccc1", "CC(=O)OC", "N#CC=C",
    "C1=CC=CC1", "C1CC=CC=C1", "C1CCC2CCCCC2C1", "[O-][N+](=O)C=C", "CC(=O)C=C", "OC(=O)C1=CCCCC1", "c1ccccc1C=C", "C1CCC1", "C=C", "CCC#N",
    "C1=CC2=CC=CC=C2C=C1", "C1CC1C1CC1", "C1CC1CC1CC1", "CC#CC=C", "C=C=C(C)C", "O=C1C=CC(=O)C=C1", "c1ccc2c(c1)CCC2", "C1CCC=CCC1", "[CH2]=C[CH2]",
    "CC(C)=C", "C1=CCC1=C", "[O-2]", "[Al+3]", "[Ti+4]", "[O-3]", "[Fe+3]", "[N-3]", "[Mg+2].[O-2]", "[Cl-].[Na+]", "[N+](=O)([O-])[O-]",
    "N#N", "[C-]#[O+]", "C#C", "CC(C)N", "CC(=O)[O-].[Na+]", "[Na+].[Cl-]",
    # isotopes, explicit hydrogens, radicals, dummies, odd valence
    "[2H]C([13CH3])O", "[13CH4]", "[13c]1ccccc1", "c1cccc[13c]1", "[14C]O", "CC[18OH]", "[3H]C",
    "[2H][2H]", "[H][H]", "[H]C([H])([H])[H]", "[H]O[H]", "[H+]", "[H]C([H])([H])N", "[2H]O[2H]",
    "[CH3]", "[CH2]", "[c]1ccccc1", "[NH2]", "[O]", "[CH]", "C[C]C", "*C", "[*]", "*c1ccccc1", "C*C",
    "[13C]", "[12C]", "[0C]", "[13CH3][13CH3]", "[2H]C#N", "CC[13C](=O)O",
    # metals and exotic elements, bracket atoms of every kind
    "[Zn+2]", "C[Hg]Cl", "[Dy+3]", "[Xe]", "F[Xe]F", "[Rh]", "[Ru+2]", "[Db]", "[Ds]", "[Re]",
    "[Fe+2]", "[Cu+]", "[Pt]", "[Ar]", "[Rn]", "[Cn]", "[Cl-]", "[Sn](C)(C)(C)C", "C[Ge](C)(C)C",
    "[Li]C", "[Mg+2]", "[Cs+]", "[Hf]", "[Ho+3]", "[Hs]", "[Rg]", "[Ra+2]", "[Rb+]", "[Rf]",
    # bonds
    "C$C", "[Mo]$[Mo]", "CC=CC", "CC#CC",
]
TARGETS += [
    # drug-like, for recursive SMARTS
    "CC(=O)Nc1ccc(O)cc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1CCC[C@H]1c1cccnc1", "NC(=O)c1ccccc1", "CCN(CC)CC", "CC(C)NCC(O)c1ccc(O)c(O)c1",
    "O=C(O)c1ccccc1O", "CS(=O)(=O)Nc1ccccc1", "NS(=O)(=O)c1ccc(N)cc1", "c1ccc(cc1)C(F)(F)F", "CC(=NO)C", "C(=O)(N)N", "OC(=O)CC(N)C(=O)O",
    "CC1=C(C(=O)N(N1C)c1ccccc1)N(C)C", "Nc1ncnc2[nH]cnc12", "O=C1NC(=O)C(=O)N1", "CC(=O)OCC", "C=CC(=O)OC", "CC(Cl)C(=O)Cl", "N#Cc1ccc(Br)cc1",
    "Oc1ccccc1", "Oc1ccccc1[N+](=O)[O-]", "COc1ccc(cc1)C=O", "CC(=O)C(C)=O", "O=C1CCCCC1", "OC1CCCCC1", "NC1CCCCC1", "CN(C)C(=O)C=C", "CCOC(=O)C1CCN(CC1)C",
    "c1ccc2c(c1)[nH]c1ccccc12", "C1CCC1C(=O)N1CCCC1", "OCC(O)CO", "CC(O)=O", "NCC(=O)NCC(=O)O", "C[n+]1ccccc1C", "CC(=O)[O-]", "CS", "CSSC", "C1CCSC1",
    "FC(F)(F)C(=O)O", "CC(C)(C)OC(=O)N", "C1=CC=C(C=C1)N=NC1=CC=CC=C1", "C=CC=CC=C", "C#CC#C", "ClCCl", "O=C=O", "NN", "ON", "NC(=N)N", "CC(=O)NN", "C1CC1C1CC1",
    "C.C", "C.O", "N.O", "CC.CC", "C.C.C", "CC.O", "[Na+].[O-]C(=O)C", "O=C1C=CC(=O)C=C1", "C1CCC2CCCCC2C1", "c1ccc2ncccc2c1", "OCCN(CCO)CCO", "[2H]C([2H])([2H])O",
    "[13CH3]C(=O)O", "C[C@H](N)C(=O)O", "FC(F)F", "CCCCC", "CCCC", "C1CCC1", "C1CC1",
]
TARGETS += [f"[{sym}]" for sym in ELEMENTS]

PROPS = "XzZhvRrx"
ELEMENT_QUERIES = [f"[{sym}]" for sym in ELEMENTS]
ELEMENT_QUERIES += [f"[{sym}{p}]" for sym in ("Dy", "Rh", "Xe", "Zn", "Hg", "Ho", "Ru", "Re", "Nh", "Cn") for p in ("", "1", "+", "H0")]
ELEMENT_QUERIES += ["[Mc]", "[Ts]", "[Og]", "[Nh]", "[Nh;h]", "[Nh1]", "[H]", "[He]", "[Hs]", "[Hf]", "[Cl]", "[Br]", "[Se]", "[se]", "[Si]", "[si]",
                    "[Te]", "[te]", "[As]", "[as]", "[B]", "[b]", "[Cc]", "[Nn]", "[Cn]", "[Sn]", "[Sc]", "[Os]", "[No]", "[Co]"]

PROP_QUERIES = []
for p in PROPS:
    PROP_QUERIES += [f"[{p}]", f"[!{p}]"]
    PROP_QUERIES += [f"[{p}{n}]" for n in range(0, 7)]
    PROP_QUERIES += [f"[!{p}{n}]" for n in (0, 1, 2, 3)]
PROP_QUERIES += ["[D]", "[D0]", "[D1]", "[D3]", "[H]", "[H0]", "[H2]", "[R300]", "[X300]", "[v300]", "[r3]", "[r4]", "[r5]", "[r6]", "[r7]", "[r8]", "[r12]"]
PROP_QUERIES += [
    "[C;R]", "[C;!R]", "[C&R0]", "[c;r6]", "[#6;R2]", "[N;X3]", "[N&X3&H0]", "[O;X2;z1]", "[C;Z1]", "[C;Z2]",
    "[C;z2]", "[c;z]", "[c;Z]", "[n;R;x2]", "[#7;h]", "[#7;!h]", "[#8;h1]", "[C;x3]", "[C;x4]", "[R,X4]", "[R;X4]",
    "[r5,r6]", "[R1;r5]", "[R2;r5]", "[x2;R1]", "[x3;R2]", "[v4;X4]", "[v4;X3]", "[v3;!R]", "[v2;R]", "[S;v6]", "[S;v4]", "[P;v5]",
    "[X1,X2]", "[!X1;!X2]", "[X3,X4;R0]", "[C;X4;R0;h0]", "[c;h0;R2]", "[C;D3;h0]", "[a;r5]", "[a;!r]", "[A;R]", "[A;!R]",
    "[Xx]", "[Dx]", "[Zz]", "[Rx]", "[Rz]", "[Hx]", "[Vv]", "[Rv]", "[nr]", "[ar]", "[cR]", "[CR]", "[nX2]", "[nH]", "[nH0]",
    "[cH]", "[cH0]", "[CH3]", "[CH2]", "[NH2]", "[NH2+]", "[OH]", "[OH0]", "[SH]", "[sH]", "[C+]", "[N+;X4]", "[N+;X3]", "[O-;X1]",
    "[C,N;R]", "[C,N;!R]", "[#6;x2;R1]", "[$([#6])]",
]
ISOTOPE_QUERIES = [
    "[13C]", "[0C]", "[12C]", "[13c]", "[13#6]", "[13]", "[0]", "[2]", "[2H]", "[H]", "[13*]", "[0*]", "[13CH4]", "[13CH3]", "[!13C]", "[!13]", "[C13]", "[C0]",
    "[13C,N]", "[13C;R]", "[13C&R]", "[2H+]", "[3H]", "[14C]", "[18O]", "[!0]", "[!0C]", "[13!C]", "[C;!13]", "[13C]O", "[13C]-[13C]",
    "[12c]", "[13N]", "[13O]", "[13S]", "[0H]", "[1H]", "[0#1]", "[0O]", "[0N]", "[#6;0]", "[0Cl]", "[13Cl]", "[0Zn]", "[2H]*", "[2H]C",
    "[0C;H3]", "[0CH3]", "[13CH3]O", "[13CH2]", "[13C;H4]", "[0#6;R]", "[0a]", "[13a]", "[0A]",
]
BOND_QUERIES = [
    "C$C", "C~C", "C-C", "C=C", "C#C", "C:C", "CC", "*$*", "[#6]$[#6]", "[Mo]$[Mo]", "A$A", "a$a", "C$[C,Mo]", "[#42]$[#42]", "[!#6]$*",
    "A", "a", "AA", "aa", "A-a", "A~a", "A=A", "a:a", "Aa", "A-A-A", "a-A", "A1AAAAA1", "a1aaaaa1", "AAA(A)A",
    "Na", "Ca", "Cn", "Cc", "Oc", "OA", "aC", "Ac", "As", "Aa", "AO",
]
NUMBER_QUERIES = ["[#0]", "[#1]", "[#6]", "[#118]", "[#119]", "[#300]", "[#1000]", "[#99999]", "[#-1]", "[#]", "[#6;R]", "[#6;!R]", "[#7;X3]"]
H_QUERIES = ["[H;R]", "[H,N]", "[!H]", "[!H;R]", "[2H;R]", "[2Hx]", "[H:1]", "[H-]", "[H+]", "[HR]", "[H*]", "[C,H]", "[C;H]", "[H1]", "[H3]", "[2H1]", "[H]C", "[H][H]", "[H]O[H]"]
MAP_QUERIES = ["[C:1]", "[#6:2]", "[C:1]-[N:2]", "[C;R:3]", "[c:1]:[c:2]"]
MISC_QUERIES = [
    "c1ccccc1", "C=O", "[#7&a]", "C#N", "[O&-]", "[C&H3&D1]-O", "c-c", "[C,N]=O", "[!#6]", "C1CC1", "[#6]-[#6;R]", "[C;R]-[C;!R]",
    "[c;R2]", "c1ccc2ccccc2c1", "[C;R1]1[C;R1][C;R1]1", "[$(C)]", "[#6;X4]", "[Q]", "[Jo]", "[Uu]", "[Xe][Xe]",
]

# Phase 2: bond expressions.
BOND_EXPR = [
    "-", "=", "#", "$", ":", "~", "@", "/", "\\", "!-", "!=", "!#", "!$", "!:", "!~", "!@", "!/", "!!-", "!!@", "-,=", "-,:", "=,#", "-,=,#", "-,:,=", "-;@", "-;!@", "-,=;@",
    "-,=;!@", "=&@", "=@", "-@", ":@", ":!@", "-!@", "=!@", "~;@", "~;!@", "~@", "~!@", "!-;@", "!-;!@", "!-&!=", "!-!=", "!-&!=&!#", "!-,!=", "!-,=", "!-;!=", "@,-", "@,=",
    "!@,-", "@;-", "@;=", "!@;-", "!@;=", "!@;:", "@;:", "-&@", "-&!@", "@&-", "!@&-", "=,#;@", "=,#;!@", "-,@", "!@,=", ",", "-,", "-&", "&-", ";-", "-;", "!", "!!", "-=", "-#", "-:", "--",
    "-@=", "/@", "\\@", "/,=", "\\,=", "/-", "-/", "=/", "!/;@", "$,@", "!$;@", "~;!$", "!~", "-,=,#,$,:", "!-&!=&!#&!$&!:", "(-,=)", "-(,=)", "-,=,", ":,=;!@", ":,-;@", "/;@", "-;-", "=;=",
]
BOND_SMARTS = []
for b in BOND_EXPR:
    BOND_SMARTS += [f"C{b}C", f"c{b}c", f"[#6]{b}[#6]", f"C{b}O", f"[#6]{b}*", f"[C;R]{b}[C;R]"]
BOND_SMARTS += [
    # in a chain, branches, rings
    "C-,=C-C", "C-,=C-,=C", "C-;!@C-;!@C", "C(-,=C)-,=C", "C(!@C)C", "C(@C)C", "C(-,=O)C", "C(=,#N)C", "C-,=C-,=C-,=C",
    "C1-,=C-C-C-C1", "C1-,=CCCC1", "C1CCCC-,=1", "C1CCCCC-,=1", "C1CCCC!@1", "C1CCCCC!@1", "C1CCCCC@1", "C@1CCCCC1", "C!@1CCCCC1", "C-1CCCCC=1", "C=1CCCCC-1", "C-1CCCCC1",
    "C-,=1CCCCC1", "C1CCCCC=,-1", "C-1CCCCC!@1", "C!@1CCCCC@1", "C%11CCCCC!@%11", "C%11CCCCC-,=%11", "C1CC1!@C", "C1CC1-;!@C", "C1CC1@C", "C1CC1!@*", "C1CC1@*", "C1CC1!@C1CC1",
    "c1ccccc1!@c1ccccc1", "c1ccccc1-;!@c1ccccc1", "c1ccccc1@c1ccccc1", "c1ccccc1-,:c1ccccc1", "c1ccccc1!:*", "c1ccc!@cc1", "c1ccc@cc1", "c1ccc:cc1", "c1ccc!:cc1", "C1CCCC=,-C1",
    "[R]!@[R]", "[R]@[R]", "[R]-;!@[R]", "[R2]!@*", "[x]!@*", "[x2]@[x2]", "*@*@*", "*!@*!@*", "*@*!@*", "[CH3]!@*", "[#6]-,=[#6;R]", "[#6;R]!@[#6;!R]", "[#6;!R]!@[#6;!R]",
    "C=,#C", "C=,#N", "C#,=N", "N=,:C", "n:c", "n!:c", "n-,:c", "c!@C", "c@c", "c:@c", "c-;!@c", "[nH]@c", "O=C!@N", "O=C@N", "O=C-,:N", "O=C1-,:N-,:C-,:C-,:C1", "OC(=O)-,=C", "[O-]-,=[N+]", "[O-]!@[N+]",
    "C-C=C", "C=C-C", "C-,=C=C", "C=C=C", "C!-C!-C", "C!-C-C", "C-!@C", "C-!@C-!@C", "C!:C!:C", "C$C", "[Mo]$[Mo]", "[Mo]!$[Mo]", "[Mo]~[Mo]", "[Mo]-,$[Mo]", "[Mo]$,=[Mo]",
    "C/C=C/C", "C/C=C\\C", "C/C", "C\\C", "C/C=C", "C/,=C", "C/&@C", "C/;@C", "C\\;!@C", "c/c", "C/c", "C!/C", "C!/;@C",
    "C1CCC2CCCCC2C1", "C1CCC2CCCCC2C1!@*", "C12CCCCC1CCCC2", "C1CC2CCC1C2", "C1CC2CCC1C2-;!@*", "C1CC2(CC1)CC2", "C1CC2(CC1)CC2!@*", "[#6]1[#6][#6][#6][#6]1",
    "C-(C)C", "C-.C", "C-", "C(-)C", "C1CCCCC=1-", "C=(C)C", "C(=)C", "C-.", "C.-C", "-C", "=C", "!C", "C1CCCCC1-", "C!-(C)C",
    "C-,=C-,=C.C-,=C", "C!@C.C@C", "C-,=C.N", "C@C.C!@C",
]

# Phase 2: ranges.
RANGE_PROPS = "DhRrvXxzZ"
RANGE_FORMS = [
    "{1-2}", "{0-1}", "{0-0}", "{1-1}", "{2-2}", "{-0}", "{-1}", "{-2}", "{-3}", "{1-}", "{2-}", "{3-}", "{0-}", "{2-1}", "{3-1}", "{1-3}", "{1-4}", "{1-8}", "{1-9}", "{0-9}", "{2-12}", "{0-20}",
    "{3-300}", "{300-400}", "{300-}", "{-300}", "{1-300}", "{0-300}", "{254-}", "{255-}", "{256-}", "{0-254}", "{0-255}", "{9-}", "{10-}", "{4-6}", "{5-7}", "{5-6}",
]
RANGE_QUERIES = []
for p in RANGE_PROPS:
    RANGE_QUERIES += [f"[{p}{r}]" for r in RANGE_FORMS]
    RANGE_QUERIES += [f"[!{p}{r}]" for r in ("{1-2}", "{-1}", "{2-}", "{0-0}", "{2-4}", "{0-300}")]
    RANGE_QUERIES += [f"[C;{p}{{1-2}}]", f"[C,N;{p}{{-2}}]", f"[{p}{{1-2}},{p}{{4-}}]", f"[{p}{{1-3}};!{p}2]", f"[{p}{{0-1}}&a]", f"[N{p}{{1-2}}]", f"[#6{p}{{1-3}}]", f"[{p}{{1-2}}C]", f"[{p}{{1-2}}{p}{{2-3}}]"]
for form in RANGE_FORMS:
    RANGE_QUERIES += [f"[+{form}]", f"[-{form}]", f"[!+{form}]", f"[!-{form}]", f"[C+{form}]", f"[N-{form}]", f"[#8-{form}]", f"[+{form};R]"]
RANGE_QUERIES += [
    "[+{1-2}+]", "[+{1-2}-]", "[+{1-2}H]", "[C+{1-2}H2]", "[+{1-2}:1]", "[R{1-2}:1]", "[R{1-2}0]", "[R{1-2}R]", "[R{1-2},X{2-}]", "[R{1-2}X{2-}]", "[D{1-2}&R{0-1}]", "[D{2-3};R{1-};!r5]",
    "[C;D{2-3};H1]", "[c;D{2-3};h{0-1}]", "[N;X{3-4};+{0-1}]", "[O;D{1-2};-{0-1}]", "[a;r{5-6}]", "[a;R{1-}]", "[x{2-4};R{1-2}]", "[C;z{1-};Z{0-1}]", "[v{3-4};!R]", "[Rx{1-2}]", "[Dx{1-2}]",
    "[H{1-2}]", "[H{-2}]", "[H{2-}]", "[D{2}]", "[D{-}]", "[D{}]", "[D{ 1-2}]", "[D{01-2}]", "[D{1-02}]", "[D{1--2}]", "[D{1-2-3}]", "[D{a-2}]", "[D{-1-2}]", "[D{+1-2}]", "[D{1-2", "[D{1-2]", "[D1-2}]", "[D{1-2}}]",
    "[D1{1-2}]", "[D{1-2}{1-2}]", "[R{1-2}{1}]", "[#6{1-2}]", "[C{1-2}]", "[Cl{1-2}]", "[^{1-2}]", "[a{1-2}]", "[A{1-2}]", "[*{1-2}]", "[13{1-2}]", "[++{1-2}]", "[--{1-2}]", "[+2{1-2}]", "[-2{1-2}]", "[+{-1-2}]", "[+{1--2}]",
    "[D{1-2147483639}]", "[D{1-2147483640}]", "[D{2147483639-}]", "[D2147483639]", "[D2147483640]", "[D4294967295]", "[D4294967296]", "[+{1-2147483640}]", "[+2147483640]", "[+2147483639]", "[+{2147483639-}]", "[-{2147483639-}]",
    "[R{12345678-}]", "[R{-12345678}]", "[D{1-1234567890}]", "[D{1-12345678901}]", "[+{00-1}]", "[+{-00}]", "[+{0-0}]", "[-{0-0}]",
    "C[D{1-2}]", "[D{1-2}]C", "[D{1-2}]-,=[D{1-2}]", "[R{1-2}]-;!@[R{0-0}]", "[R{1-}]@[R{1-}]", "[R{0-0}]!@[R{0-0}]", "[C;D{1-2}]-,=[C;D{2-3}]", "[C+{0-1}]", "[N+{0-1}]", "[N+{1-}]-,=[O-{1-}]",
]

RANGE_SMARTS = list(RANGE_QUERIES)

# Phase 3: recursive SMARTS.
RECURSIVE_ATOMS = [
    "$(C)", "$(CO)", "$(C=O)", "$(C#N)", "$(*)", "$([#6])", "$(c)", "$(a)", "$(O)", "$([O;H1])", "$(C.C)", "$(C.O)", "$(N.O)", "$(C1CC1)", "$(C1CCC1)", "$(c1ccccc1)",
    "$(cc)", "$(c:c)", "$(C-C)", "$(C-,=C)", "$(C!@C)", "$(C@C)", "$(C(=O)N)", "$(C(=O)[OH])", "$(C(=O)[O-])", "$(NC=O)", "$(N(C)C)", "$(N-C=O)", "$(N-c)", "$(O-c)",
    "$([OH]-c)", "$([OH]-C=O)", "$([NX3;H2,H1;!$(NC=O)])", "$([C;$(C=O)]N)", "$([C;$(C=O)])", "$([C;!$(C=O)]O)", "$(C=C)", "$(C=C-C=C)", "$(C-[N+](=O)[O-])", "$([N+](=O)[O-])",
    "$(S(=O)=O)", "$(S(=O)(=O)N)", "$(P=O)", "$(CC#N)", "$(C(F)(F)F)", "$(CCl)", "$(C[Br,I])", "$(*-*)", "$(*=*)", "$(*~*~*)", "$(*(~*)(~*)~*)", "$(*1~*~*~*~1)",
    "$([R])", "$([R]-[R])", "$([r5])", "$([r6]:[r6])", "$([x2])", "$([D1])", "$([D1]-[D4])", "$([D1]C)", "$([#6;D1]-[#6;D3])", "$([H])", "$([H]C)", "$([2H])", "$([13C])",
    "$([C;R]-!@C)", "$(C1CC1.C)", "$(C(C)(C)C)", "$(C(-C)(-C)-C)", "$(CCCC)", "$(CCCCC)", "$(C-,:C-,:C)", "$(c1ccc2ccccc2c1)",
    "$(C%11CC%11)", "$([C+])", "$([N+])", "$([O-])", "$([O-]C)", "$([n+][O-])", "$(c[N+]([O-])=O)", "$(cn)", "$(nc)", "$([nH])", "$(C=[N,O,S])", "$(C(=O)C=C)", "$([#6]=[#8])",
    "$([#7]-[#6]=[#8])", "$([#7;H2]-[#6])", "$([#7;H2]-[#6]=O)", "$([CH2]=[CH])", "$([CH3])", "$(C[CH3])", "$([CH3][CH3])", "$([CH3]C)",
]
RECURSIVE_QUERIES = []
for r in RECURSIVE_ATOMS:
    RECURSIVE_QUERIES += [
        f"[{r}]", f"[!{r}]", f"[C;{r}]", f"[C;!{r}]", f"[{r};C]", f"[{r}&C]", f"[N,O;{r}]", f"[#6;{r}]", f"[!{r};!#6]", f"[C,{r}]", f"[!C,{r}]", f"[{r},{r}]",
        f"C[{r}]", f"[{r}]C", f"[{r}]=O", f"[{r}]-,=C", f"O=C[{r}]", f"[{r}]-,:[{r}]", f"[{r}]!@[{r}]", f"[{r}]@[{r}]", f"[{r}]1CC1", f"C1C[{r}]1", f"[{r}]1[{r}][{r}]1",
        f"[{r}](C)C", f"C([{r}])[{r}]", f"[R;{r}]", f"[R{{1-2}};{r}]", f"[D{{1-3}};{r}]", f"[{r};D{{2-}}]", f"[{r};!R]", f"[+0;{r}]", f"[H1,{r}]", f"[{r};H0]",
    ]
RECURSIVE_QUERIES += [
    # logic between recursions
    "[$(C=O),$(C#N)]", "[$(C=O);$(CN)]", "[$(C=O)&$(CN)]", "[$(C=O)$(CN)]", "[$(C=O),$(C#N),$(C=N)]", "[!$(C=O);!$(C#N)]", "[!$(C=O),!$(C#N)]", "[!$(C=O)&!$(C#N)]",
    "[$(CO)&!$(C=O)]", "[$(CO);!$(C=O);!$(C(O)O)]", "[C;$(C=O),$(C#N)]", "[C;$(C=O);$(CN)]", "[C;$(C=O),$(C#N);!R]", "[$(C=O),C;R]", "[C,N;!$(C=O);!$(C#N)]",
    "[$(C=O),$(C#N);$(C-C)]", "[$(C=O)&$(C-C)&!$(C-N)]", "[$([#6]),$([#7])]", "[!$([#6]),$([#7])]", "[!$([#6])&!$([#7])]", "[!$([#6]);!$([#7])]", "[$(C)&$(C)]",
    # nesting
    "[$([C;$(C=O)]N)]", "[$([C;$(C=O);$(CN)]O)]", "[$([C;!$(C=O)]N)]", "[$([C,N;$(*=O)])]", "[$([$(C=O)])]", "[$([$(C=O)]N)]", "[$([$(C)]C)]", "[$([!$(C=O)]O)]",
    "[$([C;$([C;$(C=O)]N)]O)]", "[$(C($(C=O))N)]", "[$(C[$(C=O)])]", "[$(C[$(N)])]", "[$(C[$(N[$(C=O)])])]", "[$([#6;$([#6]-[#7;$([#7]-[#6]=O)])])]",
    "[$(C-[N;!$(N-C=O)])]", "[$([C;$(C-[N;!$(N-C=O)])]),$(O)]", "[$(N-C=O);$(N-[$(C=O)])]", "[$([O;$(O-[$(C=O)])])]", "[$(*-[$(*=O)])]", "[$(*-[!$(*=O)])]",
    # medchem patterns
    "[$([NX3;H2,H1;!$(NC=O)])]", "[CX3](=O)[OX2H1]", "[$([OH]-c),$([OH]-C=O)]", "[NX3;H2,H1;!$(NC=O)]", "[NX3;H2;!$(NC=O)]", "[$([NX3;H1,H2;!$(NC=O)])]C", "[CX3;$([CX3]=O)]",
    "[CX3](=O)[OX1H0-,OX2H1]", "[CX3H1](=O)[#6]", "[CX3](=[OX1])[NX3]", "[$([CX3](=O)[#6]),$([CX3](=O)[#7])]", "[OX2H][CX4;!$(C(O)(O))]", "[OX2H][$([CX4]);!$(C(O)O)]",
    "[$([NX3;!$(N-C=O);!$(N-S=O)]);H1,H2]", "[N;!$(N-[C,S]=[O,S,N])]", "[N;H0;!$(N-C=O);!$(N-S=O)]", "[nX2;!$(n:c:[n,o,s])]", "[$([nH]),$([n;H0;$(n(:c):c)])]",
    "[$([#6]=[#8]);!$([#6]-[#7,#8])]", "[$(C=O)]-[$(O-[CH3])]", "[CX4;!$(C-[!#6;!#1])]", "[CH3;$(C-[!#6])]", "[c;$(c-[N+](=O)[O-])]", "[c;$(c1ccc(cc1)[F,Cl,Br,I])]",
    "[$(c1ccccc1)]", "[$([c;R1]1[c;R1][c;R1][c;R1][c;R1][c;R1]1)]", "[O;H1;$(O-!@[#6;!$(C=*)])]", "[$([SX4](=O)(=O)([#6])[#6]),$([SX4](=O)(=O)[#6][#7])]", "[SX4](=O)(=O)[NX3;H1,H2]",
    "[$([#8]=[#6]-[#8;H1]),$([#8-]-[#6]=[#8])]", "[#7;!$([#7]-[#6]=[#8,#16,#7])]", "[#6;$([#6]=[#8]);$([#6]-[#6])]", "[CX3](=O)[$([NX3H0]),$([NX3H1]),$([NX3H2])]", "[$(*(=O)),$(*#N)]",
    "[$([Cl,Br,I])]-[$(C=O)]", "[$(C=O)]-[Cl,Br,I]", "[$([CX4][Cl,Br,I]),$(c[Cl,Br,I])]", "[$(C=C),$(C#C)]", "[OX2;$(O(C)C);!R]", "[$([OX2](C)C)]",
    # recursion plus bond logic, ranges, rings
    "[$(C=O)]-,=[$(N)]", "[$(C=O)]!@[$(N)]", "[$(C=O)]@[$(N)]", "[$(C)]-;!@[$(C)]", "[$(C)]-,=;@[$(C)]", "C-,=[$(C=O)]", "[$(C=O)]-,:[N,n]", "[$(C=O)]1CCCC1", "[$(C=O)]1CCC1",
    "[$(C=O)]1[$(N)]CC1", "[C;R;$(C=O)]", "[C;!R;$(C=O)]", "[C;R{1-2};$(C=O)]", "[C;r{5-6};$(C=C)]", "[C;D{2-3};$(C=O)]", "[C;x{2-};$(C=C)]", "[!$(C=O);R{1-}]", "[$(C=O);!D3]",
    "[$(C1CCC1)]-[$(C1CC1)]", "[$(C1CC1)]@[$(C1CC1)]", "[$(C1CC1)]!@[$(C1CC1)]", "[$(c1ccccc1)]!@[$(c1ccccc1)]", "c1ccc([$(N)])cc1", "c1ccc([$(C=O)])cc1", "c1cc[$(c-N)]cc1",
    "[c;$(c-[#8,#7])]:[c;!$(c-[#8,#7])]", "[#6;$([#6]=O)]-[#6;$([#6]=O)]", "[$(C=O)]-[$(C=O)]", "[$(C=O)]-[$(C=O)]-[$(C=O)]", "[$(C-C)]-[$(C-C)]-[$(C-C)]",
    "C=,#[$(C)]", "C1CC[$(C)]C1", "C1[$(C=O)]CCC1", "C1CCCC[$(C(=O)N)]1", "O=C1[$(N)]CCC1", "[$(C)]1[$(C)][$(C)][$(C)][$(C)][$(C)]1", "[$(C)]-,=1-,=[$(C)]-,=[$(C)]-,=[$(C)]-,=[$(C)]-,=[$(C)]-,=1",
    # recursive atom not at query start, branches, multiple components
    "CC[$(C=O)]", "C(C)([$(C=O)])N", "C(C)(C)[$(C=O)]", "N[$(C=O)]C", "N[$(C=O)].O", "N.[$(C=O)]", "[$(C=O)].[$(C=O)]", "[$(C=O)].[$(C#N)]", "[$(C)].[$(C)]", "C.[$(C)]", "[$(C)].C",
    "[$(C)][$(C)]", "[$(C)]([$(C)])[$(C)]", "[$(C)]([$(C)])([$(C)])[$(C)]", "[$(C=O)]([$(N)])[$(O)]", "[$(C=O)](N)O", "O=[$(C)]", "O=[$(C=O)]", "[$(C=O)]=O", "[$(O=C)]=C",
    # inner SMARTS features
    "[$([CH3])]", "[$([CH2][CH3])]", "[$(C(=O)[O,N])]", "[$(C(=O)([O,N])C)]", "[$(C(=O)(-[O,N])-C)]", "[$(*:*)]", "[$(c:n)]", "[$(c:[nH])]", "[$([nH]:c)]", "[$(o:c)]", "[$(s:c)]",
    "[$(C(F)(F))]", "[$([C;H0](F)(F)F)]", "[$([C;D4](F)(F)F)]", "[$(C=C=C)]", "[$(C(=C)=C)]", "[$(C#C)]", "[$(C=N-O)]", "[$(C=N-[OH])]", "[$(C(=N-O)C)]", "[$(S(=O)(=O)[O-])]",
    "[$([#8-]-[#7+]=O)]", "[$([#7+](=O)[#8-])]", "[$([#7+]([#8-])=O)]", "[$(C~C~C~C)]", "[$(C~*~C)]", "[$(C=*)]", "[$(*=*=*)]", "[$(C=C-C=C-C=C)]",
    # RDKit corner cases
    "[$(C)C]", "[C$(C)]", "[$(CC)$(CC)]", "[$(C)H]", "[H$(C)]", "[$(C)13]", "[13$(C)]", "[$(C)1]", "[$(C)@]", "[$(C)@@]", "[$(C):1]", "[$(C);R:1]", "[$([C:1])]", "[$(C=O):2]N", "[C:1][$(C=O):2]",
    "[$(C)1C]", "[$(C)12C]", "[!$(C)1]", "[!$(C)13]", "[$(C)&1]", "[$(C);1]", "[$(C),1]", "[$(C)1,N]", "[$(C)1;N]", "[$(C)1&N]", "[$(C)1H]", "[$(C)1R]", "[$(C)1:1]", "[$(C):1]C", "[$(C)C1]", "[$(C)C13]", "[$(C)#6]", "[$(C)0]",
    "[$(C)H1]", "[$(C)H2]", "[$(C)D1]", "[$(C)D{1-2}]", "[$(C)+0]", "[$(C)-]", "[$(C)+]",
]
# Whatever RDKit rejects, Rust must reject too.
RECURSIVE_BAD = [
    "[$()]", "[$(C]", "[$C]", "[$(C)(C)]", "[!$!$(C)]", "[$ (C)]", "[$(C )]", "[$( C)]", "[$(C-)]", "[$(-C)]", "[$(C.)]", "[$(C(C)]", "[$(=C)]", "[$(C:1)]", "[$($(C))]",
    "[$((C))]", "[$(C1)]", "[$(C)", "$(C)", "[$(C)]$(C)", "[$(.C)]", "[$(C..C)]", "[$(C(.C))]", "[$(CC.)]", "[$(C)))]", "[$(C]C)]", "[$([C)]", "[$(C])]", "[$(", "[$", "[$(C)(]",
    "[$(C&)]", "[$(C,)]", "[$(;C)]", "[$(C;)]", "[$(!)]", "[$(C)!]", "[$(C)&]", "[$(C)&&C]", "[$(C)$]", "[$(C)$()]", "[$($)]", "[$(C=)]", "[$(C=O]", "[$(C=O)", "[$(C(=O))]", "[$(C(=O)]",
    "C.", ".C", "C..C", "(C)C", "C(.C)", "C.(C)", "C(C).C", "C.C", "C(C)(C).C",
]
RECURSIVE_QUERIES += RECURSIVE_BAD

ALL_QUERIES = list(dict.fromkeys(
    ELEMENT_QUERIES + PROP_QUERIES + ISOTOPE_QUERIES + BOND_QUERIES + NUMBER_QUERIES + H_QUERIES + MAP_QUERIES + MISC_QUERIES
    + BOND_SMARTS + RANGE_SMARTS + RECURSIVE_QUERIES
))

# RDKit accepts these but substructure_rs deliberately does not (yet): dative bonds, and `[99999]`/`[Q]`.
UNSUPPORTED = re.compile(r"->|<-|\[99999\]|\[Q\]")


def rust_parses(path, smarts):
    pd.DataFrame({"substructure": [smarts]}).to_parquet(path, compression="zstd")
    try:
        substructure_rs.read_queries(path)
    except ValueError:
        return False
    return True


def rust_reads(path, smiles):
    pd.DataFrame({"smiles": [smiles]}).to_parquet(path, compression="zstd")
    try:
        substructure_rs.read_reactants(path)
    except ValueError:
        return False
    return True


class RdkitDifferentialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        one = tmp / "one.parquet"

        cls.rd_queries = {s: Chem.MolFromSmarts(s) for s in ALL_QUERIES}
        cls.rd_targets = {s: Chem.MolFromSmiles(s) for s in dict.fromkeys(TARGETS)}
        cls.rd_targets = {s: m for s, m in cls.rd_targets.items() if m is not None}

        cls.rust_ok = {s for s in ALL_QUERIES if rust_parses(one, s)}
        cls.rust_targets = [s for s in cls.rd_targets if rust_reads(one, s)]

        queries = [s for s in ALL_QUERIES if s in cls.rust_ok]
        cls.queries = queries
        pd.DataFrame({"substructure": queries}).to_parquet(tmp / "q.parquet", compression="zstd")
        pd.DataFrame({"smiles": cls.rust_targets}).to_parquet(tmp / "r.parquet", compression="zstd")
        index = substructure_rs.Index.build(substructure_rs.read_queries(tmp / "q.parquet"))
        found, _, _ = index.match_many(substructure_rs.read_reactants(tmp / "r.parquet"))
        cls.rust_pairs = {(cls.rust_targets[r], queries[q]) for r, qs in enumerate(found) for q in qs}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_parse_agreement(self):
        # Whatever RDKit rejects, we reject; whatever we reject must be a known gap.
        too_lax = [s for s, m in self.rd_queries.items() if m is None and s in self.rust_ok]
        self.assertEqual(too_lax, [], "accepted by Rust, rejected by RDKit")
        gaps = [s for s, m in self.rd_queries.items() if m is not None and s not in self.rust_ok and not UNSUPPORTED.search(s)]
        self.assertEqual(gaps, [], "rejected by Rust, accepted by RDKit")

    def test_targets_parse(self):
        gone = [s for s in self.rd_targets if s not in set(self.rust_targets)]
        print(f"\n{len(self.rust_targets)} targets, {len(self.queries)} queries; Rust cannot read {gone}")
        # Targets the Rust SMILES reader refuses are not this test's concern, but keep the list short.
        self.assertLess(len(gone), 12, gone)

    def test_every_pair_matches_rdkit(self):
        wrong = []
        for sma, smi in itertools.product(self.queries, self.rust_targets):
            if self.rd_queries[sma] is None:
                continue
            want = self.rd_targets[smi].HasSubstructMatch(self.rd_queries[sma])
            if want != ((smi, sma) in self.rust_pairs):
                wrong.append((sma, smi, "missing" if want else "extra"))
        total = len(self.queries) * len(self.rust_targets)
        print(f"\n{total - len(wrong)}/{total} pairs agree")
        self.assertEqual(wrong, [], f"{len(wrong)} disagreements, first: {wrong[:40]}")


if __name__ == "__main__":
    unittest.main()
