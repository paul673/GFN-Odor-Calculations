# This file contains molecule fragments which are used to construct molecules with help of the 
# Fragmented mol building environment context. This file contains three datasets. The standard 
# one and two sets with fewer fragments. Fragments not contained in molecules with the vanilla
# note in the openpom dataset have been removed in the vanilla dataset. The other dataset does
# not contain fragments not contained in the openpom dataset independent of fragance note.


# Bengio fragments used in the standard fragment mol building environment in the gflownet libary
FRAGMENTS: list[tuple[str, list[int]]] = [
    ("Br", [0]),
    ("C", [0]),
    ("C#N", [0]),
    ("C1=CCCCC1", [0, 2, 3]),
    ("C1=CNC=CC1", [0, 2]),
    ("C1CC1", [0]),
    ("C1CCCC1", [0]),
    ("C1CCCCC1", [0, 1, 2, 3, 4, 5]),
    ("C1CCNC1", [0, 2, 3, 4]),
    ("C1CCNCC1", [0, 1, 3]),
    ("C1CCOC1", [0, 1, 2, 4]),
    ("C1CCOCC1", [0, 1, 2, 4, 5]),
    ("C1CNCCN1", [2, 5]),
    ("C1COCCN1", [5]),
    ("C1COCC[NH2+]1", [5]),
    ("C=C", [0, 1]),
    ("C=C(C)C", [0]),
    ("C=CC", [0, 1]),
    ("C=N", [0]),
    ("C=O", [0]),
    ("CC", [0, 1]),
    ("CC(C)C", [1]),
    ("CC(C)O", [1]),
    ("CC(N)=O", [2]),
    ("CC=O", [1]),
    ("CCC", [1]),
    ("CCO", [1]),
    ("CN", [0, 1]),
    ("CNC", [1]),
    ("CNC(C)=O", [0]),
    ("CNC=O", [0, 2]),
    ("CO", [0, 1]),
    ("CS", [0]),
    ("C[NH3+]", [0]),
    ("C[SH2+]", [1]),
    ("Cl", [0]),
    ("F", [0]),
    ("FC(F)F", [1]),
    ("I", [0]),
    ("N", [0]),
    ("N=CN", [1]),
    ("NC=O", [0, 1]),
    ("N[SH](=O)=O", [1]),
    ("O", [0]),
    ("O=CNO", [1]),
    ("O=CO", [1]),
    ("O=C[O-]", [1]),
    ("O=PO", [1]),
    ("O=P[O-]", [1]),
    ("O=S=O", [1]),
    ("O=[NH+][O-]", [1]),
    ("O=[PH](O)O", [1]),
    ("O=[PH]([O-])O", [1]),
    ("O=[SH](=O)O", [1]),
    ("O=[SH](=O)[O-]", [1]),
    ("O=c1[nH]cnc2[nH]cnc12", [3, 6]),
    ("O=c1[nH]cnc2c1NCCN2", [8, 3]),
    ("O=c1cc[nH]c(=O)[nH]1", [2, 4]),
    ("O=c1nc2[nH]c3ccccc3nc-2c(=O)[nH]1", [8, 4, 7]),
    ("O=c1nccc[nH]1", [3, 6]),
    ("S", [0]),
    ("c1cc[nH+]cc1", [1, 3]),
    ("c1cc[nH]c1", [0, 2]),
    ("c1ccc2[nH]ccc2c1", [6]),
    ("c1ccc2ccccc2c1", [0, 2]),
    ("c1ccccc1", [0, 1, 2, 3, 4, 5]),
    ("c1ccncc1", [0, 1, 2, 4, 5]),
    ("c1ccsc1", [2, 4]),
    ("c1cn[nH]c1", [0, 1, 3, 4]),
    ("c1cncnc1", [0, 1, 3, 5]),
    ("c1cscn1", [0, 3]),
    ("c1ncc2nc[nH]c2n1", [2, 6]),
]



# Bengio fragments included in molecules from the openpom dataset with the vanilla note.
FRAGMENTS_OPENPOM_VANILLA = [
    ('C', [0]),
    ('C1=CCCCC1', [0, 2, 3]),
    ('C1CC1', [0]),
    ('C1CCCCC1', [0, 1, 2, 3, 4, 5]),
    ('C1CCOC1', [0, 1, 2, 4]),
    ('C1CCOCC1', [0, 1, 2, 4, 5]),
    ('C=C', [0, 1]),
    ('C=C(C)C', [0]),
    ('C=CC', [0, 1]),
    ('C=O', [0]),
    ('CC', [0, 1]),
    ('CC(C)C', [1]),
    ('CC(C)O', [1]),
    ('CC=O', [1]),
    ('CCC', [1]),
    ('CCO', [1]),
    ('CO', [0, 1]),
    ('CS', [0]),
    ('N', [0]),
    ('O', [0]),
    ('O=CO', [1]),
    ('O=C[O-]', [1]),
    ('S', [0]),
    ('c1cc[nH]c1', [0, 2]),
    ('c1ccccc1', [0, 1, 2, 3, 4, 5]),
    ('c1ccsc1', [2, 4])
    ]

# Bengio fragments included in molecules from the openpom dataset.
FRAGMENTS_OPENPOM_DATASET = [
    ('Br', [0]),
    ('C', [0]),
    ('C#N', [0]),
    ('C1=CCCCC1', [0, 2, 3]),
    ('C1CC1', [0]),
    ('C1CCCC1', [0]),
    ('C1CCCCC1', [0, 1, 2, 3, 4, 5]),
    ('C1CCNC1', [0, 2, 3, 4]),
    ('C1CCNCC1', [0, 1, 3]),
    ('C1CCOC1', [0, 1, 2, 4]),
    ('C1CCOCC1', [0, 1, 2, 4, 5]),
    ('C1CNCCN1', [2, 5]),
    ('C=C', [0, 1]),
    ('C=C(C)C', [0]),
    ('C=CC', [0, 1]),
    ('C=N', [0]),
    ('C=O', [0]),
    ('CC', [0, 1]),
    ('CC(C)C', [1]),
    ('CC(C)O', [1]),
    ('CC(N)=O', [2]),
    ('CC=O', [1]),
    ('CCC', [1]),
    ('CCO', [1]),
    ('CN', [0, 1]),
    ('CNC', [1]),
    ('CNC(C)=O', [0]),
    ('CNC=O', [0, 2]),
    ('CO', [0, 1]),
    ('CS', [0]),
    ('C[NH3+]', [0]),
    ('C[SH2+]', [1]),
    ('Cl', [0]),
    ('F', [0]),
    ('I', [0]),
    ('N', [0]),
    ('N=CN', [1]),
    ('NC=O', [0, 1]),
    ('N[SH](=O)=O', [1]),
    ('O', [0]),
    ('O=CO', [1]),
    ('O=C[O-]', [1]),
    ('O=PO', [1]),
    ('O=P[O-]', [1]),
    ('O=S=O', [1]),
    ('O=[NH+][O-]', [1]),
    ('O=[PH](O)O', [1]),
    ('O=[PH]([O-])O', [1]),
    ('O=[SH](=O)O', [1]),
    ('O=[SH](=O)[O-]', [1]),
    ('O=c1[nH]cnc2[nH]cnc12', [3, 6]),
    ('O=c1[nH]cnc2c1NCCN2', [8, 3]),
    ('O=c1cc[nH]c(=O)[nH]1', [2, 4]),
    ('O=c1nccc[nH]1', [3, 6]),
    ('S', [0]),
    ('c1cc[nH]c1', [0, 2]),
    ('c1ccc2[nH]ccc2c1', [6]),
    ('c1ccc2ccccc2c1', [0, 2]),
    ('c1ccccc1', [0, 1, 2, 3, 4, 5]),
    ('c1ccncc1', [0, 1, 2, 4, 5]),
    ('c1ccsc1', [2, 4]),
    ('c1cn[nH]c1', [0, 1, 3, 4]),
    ('c1cncnc1', [0, 1, 3, 5]),
    ('c1cscn1', [0, 3]),
    ('c1ncc2nc[nH]c2n1', [2, 6])
    ]

