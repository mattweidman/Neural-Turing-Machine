import numpy as np
from scipy import ndimage

epsilon = 0.0000001

def softmax(u):
    u = u - u.mean()
    exp_u = np.exp(u)
    return exp_u / exp_u.sum()

# find the numerical gradient of function f with respect to param
def numerical_gradient_matrix(f, param):
    grad_param = np.zeros(param.shape)
    for i in range(len(param)):
        for j in range(len(param[i])):
            param[i,j] += epsilon
            f1 = f().sum()
            param[i,j] -= 2*epsilon
            f2 = f().sum()
            param[i,j] += epsilon
            grad_param[i,j] = (f1-f2)/(2*epsilon)
    return grad_param

# find the numerical gradient of function f with respect to param
def numerical_gradient_array(f, param):
    grad_param = np.zeros(param.shape)
    for i in range(len(param)):
        param[i] += epsilon
        f1 = f()[i]
        param[i] -= 2*epsilon
        f2 = f()[i]
        param[i] += epsilon
        grad_param[i] = (f1-f2)/(2*epsilon)
    return grad_param

# find the numerical gradient of function f with respect to param
# f needs to take param as an input
def numerical_gradient_scalar(f, param):
    param += epsilon
    f1 = f(param)
    param -= 2*epsilon
    f2 = f(param)
    param += epsilon
    return (f1-f2)/(2*epsilon)

def check_numerical_gradient():
    a = np.array([1.0, 2.0])
    f = lambda: a**2
    n_grad = numerical_gradient_array(f, a)
    grad = 2*a
    diff = n_grad-grad
    print(diff)

def check_numerical_gradient_scalar():
    a = np.random.randn(10)
    x = np.random.randn()
    f = lambda x: x**2
    n_grad = numerical_gradient_scalar(f, x)
    grad = 2*x
    print(n_grad-grad)

def check_w_tilde():
    n = 10
    w = softmax(np.random.randn(n))
    gamma = np.random.randn()
    f = lambda: w**gamma / (w**gamma).sum()
    n_grad = numerical_gradient_array(f, w)
    wg = w**gamma
    wg1 = w**(gamma-1)
    grad = gamma * wg1 / wg.sum() * (1-wg/wg.sum())
    diff = n_grad - grad
    print(diff)

def check_gamma():
    n = 10
    w = softmax(np.random.randn(n))
    gamma = np.random.randn()
    f = lambda gamma: w**gamma / (w**gamma).sum()
    n_grad = numerical_gradient_scalar(f, gamma)
    wg = w**gamma
    grad = wg/wg.sum()**2 * (np.log(w)*wg.sum() - np.sum(wg*np.log(w)))
    print(grad-n_grad)

def circular_convolve(a, b):
    return np.roll(ndimage.convolve(a, b, mode='wrap'), len(a)//2)

def check_convolve():
    n = 9
    wt = softmax(np.random.randn(n))
    s = softmax(np.random.randn(n))
    f = lambda: circular_convolve(wt, s)

    # gradient with respect to wt
    n_grad = numerical_gradient_array(f, wt)
    grad = s[0] * np.ones(s.shape)
    diff = n_grad - grad
    print(diff)

    # gradient with respect to s
    n_grad = numerical_gradient_array(f, s)
    grad = wt[0] * np.ones(wt.shape)
    diff = n_grad - grad
    print(diff)

def check_gated():
    n = 9
    wc = softmax(np.random.randn(n))
    w_prev = softmax(np.random.randn(n))
    g = np.random.randn()
    f = lambda: g*wc + (1-g)*w_prev

    # gradient with respect to wc
    n_grad = numerical_gradient_array(f, wc)
    grad = g * np.ones(wc.shape)
    print(n_grad-grad)

    # gradient with respect to w_prev
    n_grad = numerical_gradient_array(f, w_prev)
    grad = (1-g) * np.ones(w_prev.shape)
    print(n_grad-grad)

    # gradient with respect to g
    f = lambda g: g*wc + (1-g)*w_prev
    n_grad = numerical_gradient_scalar(f, g)
    grad = wc - w_prev
    print(n_grad - grad)

def check_softmax():
    n = 9
    x = softmax(np.random.randn(n))
    f = lambda: softmax(x)
    n_grad = numerical_gradient_array(f, x)
    grad = f() * (1-f())
    print(n_grad-grad)

def check_key():
    n = 9
    m = 10
    beta = np.random.randn()
    k = np.random.randn(m)
    M = np.random.randn(n, m)
    f = lambda: beta * k[np.newaxis,:].dot(M.T) / (np.linalg.norm(k) *
        np.linalg.norm(M, axis=1))

    # numerical gradient with respect to k
    n_grad = np.zeros((n, m))
    for j in range(m):
        k[j] += epsilon
        f1 = f()
        k[j] -= 2*epsilon
        f2 = f()
        k[j] += epsilon
        n_grad[:,j] = (f1-f2)/(2*epsilon)

    # gradient with respect to k
    grad = np.zeros((n,m))
    k_norm = np.sqrt((k**2).sum())
    for i in range(n):
        Mi_norm = np.sqrt((M[i]**2).sum())
        for j in range(m):
            grad[i,j] = beta / (k_norm * Mi_norm) \
                * (M[i,j] - k[j]*k.dot(M[i])/k_norm**2)
    # grad = 1/norm_k * (M/norm_M).sum(axis=0) - k / norm_k**3 * \
    #     k.dot((M/norm_M).T).sum()
    # grad = beta * (M/np.linalg.norm(M, axis=1)[:,np.newaxis]).sum(axis=0)
    print(n_grad - grad)

if __name__ == "__main__":
    check_key()
