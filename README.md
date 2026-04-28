# 3x3 matmul, rank 23, 56 additions

A self-contained certificate for a non-commutative 3x3 matrix multiplication algorithm with 23 bilinear products and 56 scalar additions (U=13, V=13, W=30). `verify.py` inlines the straight-line program, reconstructs the U/V/W factors, and runs: Brent's identity over Z, GF(2), GF(3); CSE self-consistency (ternary entries, declared addition counts); 1000 random factor trials per modulus against `numpy`'s `A @ B`; 200 non-commutative trials with 2x2 matrix entries; and an input-side optimality lower bound certifying U and V each require >=13 additions. Run `python3 verify.py`. 

