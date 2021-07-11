import numpy as np


def linear_cca(H1, H2, cat, use_cluster, outdim_size, beta):
    """
    An implementation of linear CCA
    # Arguments:
        H1 and H2: the matrices containing the data for view 1 and view 2. Each row is a sample.
        outdim_size: specifies the number of new features
    # Returns
        A and B: the linear transformation matrices 
        mean1 and mean2: the means of data for both views
    """
    r1 = 1e-4
    r2 = 1e-4

    m = H1.shape[0]
    o = H1.shape[1]

    mean1 = np.mean(H1, axis=0)
    mean2 = np.mean(H2, axis=0)
    H1bar = H1 - np.tile(mean1, (m, 1))
    H2bar = H2 - np.tile(mean2, (m, 1))

    SigmaHat11 = (1.0 / (m - 1)) * np.dot(H1bar.T, H1bar) + r1 * np.identity(H1.shape[1])
    SigmaHat22 = (1.0 / (m - 1)) * np.dot(H2bar.T, H2bar) + r2 * np.identity(H2.shape[1])

    if (use_cluster):
        FEATDIM = o
        NUMCAT  = 10
        corr = np.zeros([FEATDIM,FEATDIM], dtype=np.float32)
        num  = 0
        for c in range(NUMCAT):
            flag = (cat==c)
            xi = H1bar[flag,:]     # use centered value
            xj = H2bar[flag,:]
            size = xi.shape
            # for ni in range(size[0]):
            #     xiext = np.tile(np.transpose(xi[ni:ni+1]), (1,size[0]))
            #     corr += np.matmul(xiext, xj)
            #     num += size[0]
            xiext= np.repeat(xi, [size[0]], axis=0)
            xjext= np.tile(xj, (size[0], 1))
            corr += np.matmul(np.transpose(xiext), xjext)
            num += size[0]*size[0]

        SigmaHat12_c = corr/num
        SigmaHat12_r = (1.0 / (m - 1)) * np.dot(H1bar.T, H2bar)
        SigmaHat12 = beta * SigmaHat12_r + (1-beta) * SigmaHat12_c
    else:
        SigmaHat12 = (1.0 / (m - 1)) * np.dot(H1bar.T, H2bar)


    [D1, V1] = np.linalg.eig(SigmaHat11)
    [D2, V2] = np.linalg.eig(SigmaHat22)
    SigmaHat11RootInv = np.dot(np.dot(V1, np.diag(D1 ** -0.5)), V1.T)
    SigmaHat22RootInv = np.dot(np.dot(V2, np.diag(D2 ** -0.5)), V2.T)

    Tval = np.dot(np.dot(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

    [U, D, V] = np.linalg.svd(Tval)
    V = V.T
    A = np.dot(SigmaHat11RootInv, U[:, 0:outdim_size])
    B = np.dot(SigmaHat22RootInv, V[:, 0:outdim_size])
    D = D[0:outdim_size]

    return A, B, mean1, mean2
