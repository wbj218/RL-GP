from kernel import kernel
import numpy as np
from numpy.linalg import det


def hellinger_distance(mu, sigma, mu_full, sigma_full):
    sigma_bar = (sigma + sigma_full)/2
    u = mu - mu_full
    d = ((det(sigma)**0.25)*(det(sigma_full)**0.25)/(det(sigma_bar)**0.5)) * np.exp(-0.125*(u.dot(np.linalg.inv(sigma_bar)).dot(u)))
    return np.sqrt(1-d)

def dhmp(D, y, x, theta, keps, noise_prior, xvec = False):
    KDD = kernel(theta, D, D, X_vector=xvec, Y_vector=xvec)
    kxx = kernel(theta, x, x, X_vector=True, Y_vector=True)
    KDx = kernel(theta, D, x, X_vector=xvec, Y_vector=True)
    B = np.eye(y.shape[1])
    KDD = np.kron(KDD, B)
    kxx = np.kron(kxx, B)
    KDx = np.kron(KDx, B)
    y_flatten = y.flatten()

    mu_full = KDx.T.dot(np.linalg.inv(KDD + np.kron(noise_prior**2, np.eye(D.shape[0])))).dot(y_flatten)
    sigma_full = kxx - KDx.T.dot(np.linalg.inv(KDD + np.kron(noise_prior**2, np.eye(D.shape[0])))).dot(KDx) + noise_prior**2

    #remaining indice set to consider
    I = np.array(range(D.shape[0]))
    II = np.array(range(D.shape[0], 2*D.shape[0]))
    II = np.concatenate([I,II])

    prune = True

    while prune:
        if len(I) == 1:
            prune = False
            break

        gmin = np.Inf

        for i in I:
            remove = np.array([2*i, 2*i+1])
            remain = np.setdiff1d(II, remove)
            KDx_remain = KDx[remain]
            y_remain = y_flatten[remain]

            remain = np.setdiff1d(I,i)
            D_remain = D[remain]
            if len(remain) == 1:
                ve = True
            else:   
                ve = False
            KDD_remain = kernel(theta, D_remain, D_remain, X_vector=ve, Y_vector=ve)
            KDD_remain = np.kron(KDD_remain, B)

            mu = KDx_remain.T.dot(np.linalg.inv(KDD_remain + np.kron(noise_prior**2, np.eye(D_remain.shape[0])))).dot(y_remain)
            sigma = kxx - KDx_remain.T.dot(np.linalg.inv(KDD_remain + np.kron(noise_prior**2, np.eye(D_remain.shape[0])))).dot(KDx_remain) + noise_prior**2
            
            d = hellinger_distance(mu, sigma, mu_full, sigma_full)

            if d < gmin:
                gmin = d
                imin = i

        if gmin > keps:
            prune = False

        else:
            #print("Prune!")
            I = np.setdiff1d(I, imin)
            rem = np.array([2*imin, 2*imin+1])
            II = np.setdiff1d(II, rem)

    return I

    

