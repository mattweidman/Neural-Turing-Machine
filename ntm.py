import numpy as np

from LSTM import LSTM

# finds the cosine of the angle between vectors in K and vectors in M
# k: size (num_examples, M)
# M: size (N, M)
# return: size (num_examples, N)
def K(k, M):
    k = k[np.newaxis,:]
    return k.dot(M.T)[0] / (np.linalg.norm(k) * np.linalg.norm(M, axis=1))

def softmax(u):
    u = u - u.mean()
    exp_u = np.exp(u)
    return exp_u / exp_u.sum()

# neural turing machine with LSTM controller
# NOTE: using more than one example at a time does not work yet
class NTM:

    # N: number of rows of memory
    # M: number of columns of memory
    # R: number of read heads
    # W: number of write heads
    # X_size: size of output
    # Y_size: size of input
    # hidden_sizes: list of number of neurons for the 1..n-1 layers (input
    # and output sizes not included)
    def __init__(self, N, M, R, W, X_size, Y_size, hidden_sizes):
        self.N = N
        self.M = M
        self.R = R
        self.W = W
        self.X_size = X_size
        self.Y_size = Y_size
        self.c_input_size = X_size + R*M
        self.c_output_size = Y_size + (R+W)*(M+N+3) + 2*W*M
        self.layer_sizes = [self.c_input_size] + hidden_sizes + \
            [self.c_output_size]
        self.lstm = LSTM(self.layer_sizes)
        self.memory = np.random.randn(N, M)

    # X: input, size (X_size)
    # r: last vectors read from memory, size (R, M)
    # s_prev, h_prev: inputs from previous forward_prop_lstm_once, each a list
    # of (layer_output_size) matrices
    # return: (s, h, gates, outp, read_heads, write_heads, add_vec, erase_vec)
    # s, h, and gates are the output from LSTM.forward_prop_once()
    # outp: output, size (Y_size)
    # read_heads: size (R, M+N+3)
    # write_heads: size (W, M+N+3)
    # add_vec: size (W, M)
    # erase_vec: size (W, M)
    def forward_prop_lstm_once(self, X, r, s_prev, h_prev):

        # forward prop LSTM
        r = r.reshape(self.R*self.M)
        controller_input = np.concatenate((X, r))[np.newaxis,:]
        s, h, gates = self.lstm.forward_prop_once(controller_input, s_prev,
            h_prev, return_gates=True)
        contr_output = h[-1][0]

        # NTM output
        outp = contr_output[:self.Y_size]

        # read heads
        rw_index = self.Y_size + self.R*(self.M+self.N+3)
        read_heads = contr_output[self.Y_size:rw_index]
        read_heads = read_heads.reshape(self.R, self.M+self.N+3)

        # write heads
        we_index = rw_index + self.W*(self.M+self.N+3)
        write_heads = contr_output[rw_index:we_index]
        write_heads = write_heads.reshape(self.W, self.M+self.N+3)

        # add vector
        ea_index = we_index + self.W*self.M
        add_vec = contr_output[we_index:ea_index]
        add_vec = add_vec.reshape(self.W, self.M)

        # erase vector
        erase_vec = contr_output[ea_index:]
        erase_vec = erase_vec.reshape(self.W, self.M)

        return s, h, gates, outp, read_heads, write_heads, add_vec, erase_vec

    # computes next memory-indexing weights for a read or write head
    # w_prev: previous memory-indexing weights, size (N)
    # all elements of w_prev must be between 0 and 1 and the sum must be 1
    # k: key vector, size (M)
    # beta: key strength, scalar
    # g: interpolation gate, scalar; all elements of g must be between 0 and 1
    # s: shift vector, size (N)
    # all elements of s should be between 0 and 1 and the sum bust be 1
    # gamma: sharpness, scalar
    def compute_w(self, w_prev, k, beta, g, s, gamma):
        wc = softmax(beta * K(k, self.memory))
        wg = g*wc + (1-g)*w_prev
        wt = np.convolve(wg, s, "same")
        wtgamma = wt ** gamma
        w = wtgamma / wtgamma.sum()
        return w

    '''# modifies memory and computes next output
    # X: input, size (num_examples, X_size)
    # r: last reads from memory, size (num_examples, R*M)
    # s_prev: previous LSTM internal state, size (num_examples, M+N+3)
    # h_prev: previous LSTM output, size (num_examples, M+N+3)
    # wr_prev: previous read indexing matrices, size (num_examples, R, N)
    # ww_prev: previous write indexing matrices, size (num_examples, W, N)
    def forward_prop_once(self, X, r, s_prev, h_prev, wr_prev, ww_prev):

        num_examples = X.shape[0]

        # forward prop once
        s, h, gates, outp, read_heads, write_heads, add_vec, erase_vec = \
            forward_prop_lstm_once(X, r, s_prev, h_prev)

        # read from memory
        read_weights = []
        for i in range(self.R):
            w = compute_w(w_prev[:,i,:], read_heads[:,:self.M],
                read_heads[:,self.M], read_heads[:,self.M+1],
                read_heads[:,self.M+1:self.M+self.N+1], read_heads[:,-1])
            read_weights.append(w)
        read_weights = np.concatenate(read_weights, axis=1).swapaxes(0,1)

        # write to memory
        for i in range(self.W):
            w = compute_w(w_prev[:,i,:], write_heads[:,:self.M],
                write_heads[:,self.M], write_heads[:,self.M+1],
                write_heads[:,self.M+1:self.M+self.N+1], write_heads[:,-1])
            we = w.dot(erase_vec)'''
