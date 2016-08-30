import numpy as np

from ntm import NTM
from ntm import softmax

def test_forward_prop_lstm_once():
    # parameters
    N = 10
    M = 15
    R = 4
    W = 3
    X_size = 12
    Y_size = 5
    hidden_size = 8

    # construct NTM and inputs
    ntm = NTM(N, M, R, W, X_size, Y_size, [hidden_size])
    X = np.random.randn(X_size)
    r = np.random.randn(R, M)
    s_prev = [np.random.randn(1, l.output_size) for l in ntm.lstm.layers]
    h_prev = [np.random.randn(1, l.output_size) for l in ntm.lstm.layers]
    out_tuple = ntm.forward_prop_lstm_once(X, r, s_prev, h_prev)
    for w in out_tuple[3:]:
        print(np.linalg.norm(w))

def test_compute_w():
    # parameters
    N = 10
    M = 15
    R = 4
    W = 3
    X_size = 12
    Y_size = 5
    hidden_size = 8

    # construct NTM and inputs
    ntm = NTM(N, M, R, W, X_size, Y_size, [hidden_size])
    w_prev = softmax(np.fabs(np.random.randn(N)))
    k = np.random.randn(M)
    beta = np.random.randn()
    g = 1/(1+np.exp(-np.random.randn()))
    s = softmax(np.random.randn(N))
    gamma = np.random.randn()
    w = ntm.compute_w(w_prev, k, beta, g, s, gamma)
    print(w)

def test_compute_w():
    # parameters
    N = 10
    M = 15
    R = 4
    W = 3
    X_size = 12
    Y_size = 5
    hidden_size = 8

    # construct NTM and inputs
    ntm = NTM(N, M, R, W, X_size, Y_size, [hidden_size])
    X = np.random.randn(X_size)
    r = np.random.randn(R, M)
    s_prev = [np.random.randn(1, l.output_size) for l in ntm.lstm.layers]
    h_prev = [np.random.randn(1, l.output_size) for l in ntm.lstm.layers]
    wr_prev = softmax(np.random.randn(R, N))
    ww_prev = softmax(np.random.randn(W, N))
    read_weights, outp = ntm.forward_prop_once(X, r, s_prev, h_prev, wr_prev,
        ww_prev)
    print(read_weights)
    print(outp)

if __name__ == "__main__":
    test_compute_w()
