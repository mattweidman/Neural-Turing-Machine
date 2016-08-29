import numpy as np

from LSTM import LSTM

# neural turing machine with LSTM controller
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
        self.memory = np.zeros((N, M))

    # finds the cosine of the angle between 2 vectors
    def cos_angle(u, v):
        return u.dot(v)/(np.linalg.norm(u)*np.linalg.norm(v))

    # X: input, size (num_examples, X_size)
    # r: last vectors read from memory, size (num_examples, R*M)
    # s_prev, h_prev: inputs from previous forward_prop_lstm_once, each size
    # (num_examples, controller_output_size)
    # return: (s, h, gates, outp, read_heads, write_heads, add_vec, erase_vec)
    # s, h, and gates are the output from LSTM.forward_prop_once()
    # outp: output, size (num_examples, Y_size)
    # read_heads: size (num_examples, R, M+N+3)
    # write_heads: size (num_examples, W, M+N+3)
    # add_vec: size (num_examples, W, M)
    # erase_vec: size (num_examples, W, M)
    def forward_prop_lstm_once(self, X, r, s_prev, h_prev):
        num_examples = X.shape[0]

        # forward prop LSTM
        controller_input = np.concatenate((X, r), axis=1)
        s, h, gates = self.lstm.forward_prop_once(controller_input, s_prev,
            h_prev, return_gates=True)
        contr_output = h[-1]

        # NTM output
        outp = contr_output[:,:self.Y_size]

        # read heads
        rw_index = self.Y_size + self.R*(self.M+self.N+3)
        read_heads = contr_output[:, self.Y_size:rw_index]
        read_heads = read_heads.reshape(num_examples, self.R, self.M+self.N+3)

        # write heads
        we_index = rw_index + self.W*(self.M+self.N+3)
        write_heads = contr_output[:, rw_index:we_index]
        write_heads = write_heads.reshape(num_examples, self.W, self.M+self.N+3)

        # add vector
        ea_index = we_index + self.W*self.M
        add_vec = contr_output[:, we_index:ea_index]
        add_vec = add_vec.reshape(num_examples, self.W, self.M)

        # erase vector
        erase_vec = contr_output[:, ea_index:]
        erase_vec = erase_vec.reshape(num_examples, self.W, self.M)

        return s, h, gates, outp, read_heads, write_heads, add_vec, erase_vec
