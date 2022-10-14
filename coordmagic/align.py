import numpy as np

def kabsch_rmsd(st1, st2, weight = None) -> float:
    """
    Rotate matrix st1.coord  onto st2.coord using Kabsch algorithm and calculate the RMSD.
    An optional vector of weights W may be provided.
    Parameters
    ----------
    st1 : coordmagic structure object 1
    st2 : coordmagic structure object 2
    W : array of length of atoms in st
    Returns
    -------
    rmsd : float
        root-mean squared deviation
    """
    coord1 = np.array(st1.coord)
    coord2 = np.array(st2.coord)
    if weight is None:
        weight = np.ones(len(coord1)) / len(coord1)
    # Computation of the weighted covariance matrix
    C = np.zeros((3, 3))
    W = np.array([weight, weight, weight]).T
    # NOTE UNUSED psq = 0.0
    # NOTE UNUSED qsq = 0.0
    iw = 3.0 / W.sum()
    n = len(coord1)
    for i in range(3):
        for j in range(n):
            for k in range(3):
                C[i, k] += coord1[j, i] * coord2[j, k] * W[j, i]
    CMP = (coord1 * W).sum(axis=0)
    CMQ = (coord2 * W).sum(axis=0)
    PSQ = (coord1 * coord1 * W).sum() - (CMP * CMP).sum() * iw
    QSQ = (coord2 * coord2 * W).sum() - (CMQ * CMQ).sum() * iw
    C = (C - np.outer(CMP, CMQ) * iw) * iw

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    if  (np.linalg.det(V) * np.linalg.det(W)) < 0.0:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    # Create Rotation matrix U, translation vector V, and calculate RMSD:
    U = np.dot(V, W)
    msd = (PSQ + QSQ) * iw - 2.0 * S.sum()
    if msd < 0.0:
        msd = 0.0
    rmsd = np.sqrt(msd)
    V = np.zeros(3)
    for i in range(3):
        t = (U[i, :] * CMQ).sum()
        V[i] = CMP[i] - t
    V = V * iw
    return U, V, rmsd


