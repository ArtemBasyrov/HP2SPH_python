"""Structure of the alias fold, which is what makes the folded system hard.

Three facts, all measured, that between them close the low-rank and the
block-Toeplitz routes:
  * the fold touches about half the latitude rows, so it is NOT a low-rank update;
  * the number of coupled longitude-column pairs GROWS with resolution, so a
    block-Toeplitz matvec gets worse, not better;
  * the column-coupling graph splits into exactly 4 components of size nside.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from src.double_fourier_sphere import dfs_fold_sparse

print(
    f"{'nside':>6} {'rows':>6} {'cols':>6} {'relaxed':>9} {'%':>6} "
    f"{'rows hit':>9} {'%':>6} {'col pairs':>10} {'/n_trans':>9} {'components':>11} {'size':>6}"
)
for nside in (16, 32, 64, 128, 256):
    p = dfs_fold_sparse(nside, 2, 1e-2)
    Mh, nt = p.n_rows, p.n_trans
    rows = np.unique(p.src % Mh)
    sc, dc = p.src // Mh, p.dst // Mh
    pairs = np.unique(np.stack([sc, dc]), axis=1).shape[1]
    ncomp, lab = connected_components(
        coo_matrix((np.ones(sc.size), (sc, dc)), shape=(nt, nt)), directed=False
    )
    sizes = np.bincount(lab)
    print(
        f"{nside:6d} {Mh:6d} {nt:6d} {p.src.size:9d} "
        f"{100 * p.src.size / (Mh * nt):5.2f}% {rows.size:9d} "
        f"{100 * rows.size / Mh:5.1f}% {pairs:10d} {pairs / nt:8.2f}x "
        f"{ncomp:11d} {sizes.max():6d}"
    )
