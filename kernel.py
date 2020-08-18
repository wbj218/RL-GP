import numpy as np

def kernel(theta, X, Y, X_vector=False, Y_vector=False):
    if X_vector:
        X = X.reshape(1,-1)
        n1 = X.shape[0]
    else:
        n1 = X.shape[0]
    if Y_vector:
        Y = Y.reshape(1,-1)
        n2 = Y.shape[0]
    else:
        n2 = Y.shape[0]

    b = theta[0:-1]
    b = b.reshape([1,-1])
    c = theta[-1]

    X = X*np.repeat(np.sqrt(b), n1, axis=0)
    Y = Y*np.repeat(np.sqrt(b), n2, axis=0)

    K = -2*np.matmul(X, Y.T) + np.repeat(np.sum(Y*Y, axis=1).reshape([1,-1]), n1, axis=0) + \
        np.repeat(np.sum(X*X, axis=1).reshape([-1,1]), n2, axis=1)
    K = c*np.exp(-0.5*K)

    return K





