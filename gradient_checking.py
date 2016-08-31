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
    wt = softmax(np.random.randn(n))
    gamma = np.random.randn()
    f = lambda: wt**gamma / (wt**gamma).sum()
    n_grad = numerical_gradient_array(f, wt)
    wgamma = wt**gamma
    wgsum = wgamma.sum()
    grad = gamma * (wt**(gamma-1))/wgsum * (1-wgamma/wgsum)
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
    x = 1/(1+np.exp(-np.random.randn(n)))
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
    k = np.random.randn(1, m)
    M = np.random.randn(n, m)
    f = lambda: beta*k.dot(M.T) / (np.linalg.norm(k)*np.linalg.norm(M, axis=1))

    # gradient with respect to k
    n_grad = numerical_gradient_matrix(f, k)
    k_norm = np.sqrt((k**2).sum())
    M_norm = np.sqrt((M**2).sum(axis=1))[:,np.newaxis]
    grad = beta / (k_norm * M_norm) * (M - M.dot(k.T).dot(k)/k_norm**2)
    grad = grad.sum(axis=0)
    print(n_grad - grad)

    # gradient with respect to M
    n_grad = numerical_gradient_matrix(f, M)
    grad = beta / (k_norm * M_norm) * (k - M * M.dot(k.T) / M_norm**2)
    print(n_grad-grad)

def compute_w_last2(wg, s, gamma):
    wt = np.zeros(wg.shape)
    for i in range(len(wg)):
        for j in range(len(s)):
            wt[i] += wg[j] * s[i-j]
    w = wt**gamma / (wt**gamma).sum()
    return wt, w

def backprop_w_last2(wg, s, gamma):
    wt = compute_w_last2(wg, s, gamma)[0]
    wgamma = wt**gamma
    wgsum = wgamma.sum()
    dwt = gamma * (wt**(gamma-1))/wgsum * (1-wgamma/wgsum)
    dgamma = wgamma/wgsum**2 * (np.log(wt)*wgsum -
        np.sum(wgamma*np.log(wt)))
    dwg = np.zeros(wg.shape)
    for i in range(len(wg)):
        for j in range(len(wt)):
            dwg[i] += dwt[j] * s[j-i]
    return dgamma, dwg

def check_w_last2():
    n = 9
    wg = softmax(np.random.randn(n))
    s = softmax(np.random.randn(n))
    gamma = 1/(1+np.exp(-np.random.randn()))
    grad = backprop_w_last2(wg, s, gamma)

    # check gamma
    f = lambda gamma: compute_w_last2(wg, s, gamma)[1]
    n_grad = numerical_gradient_scalar(f, gamma)
    print(n_grad - grad[0])

    # check wg
    f = lambda: compute_w_last2(wg, s, gamma)[0]
    n_grad = numerical_gradient_array(f, wg)
    print(n_grad)
    print(grad[1])
    print(n_grad - grad[1])

def compute_w(w_prev, M_prev, k, beta, g, s, gamma):
    u = beta * k.dot(M_prev.T)/(np.linalg.norm(k)*np.linalg.norm(M_prev, axis=1))
    wc = softmax(u)
    wg = g*wc + (1-g)*w_prev
    wt = circular_convolve(wg, s)
    w = wt**gamma / (wt**gamma).sum()
    return u, wc, wg, wt, w

def backprop_w(w_prev, M_prev, k, beta, g, s, gamma, u, wc, wg, wt, grad_in):
    wgamma = wt**gamma
    wgsum = wgamma.sum()
    dwt = grad_in * gamma * (wt**(gamma-1))/wgsum * (1-wgamma/wgsum)
    dgamma = grad_in * wgamma/wgsum**2 * (np.log(wt)*wgsum -
        np.sum(wgamma*np.log(wt)))
    dwg = dwt * s[0] * np.ones(wg.shape)
    ds = dwt * wg[0] * np.ones(s.shape)
    dwc = dwg * g * np.ones(wc.shape)
    dw_prev = dwg * (1-g) * np.ones(w_prev.shape)
    dg = dwg * (wc - w_prev)
    du = dwc * u * (1-u)
    dbeta = du * u / beta
    k = k[np.newaxis,:]
    k_norm = np.sqrt((k**2).sum())
    M_norm = np.sqrt((M_prev**2).sum(axis=1))[:,np.newaxis]
    dk = du[:,np.newaxis] * beta / (k_norm * M_norm) * (M_prev -
        M_prev.dot(k.T).dot(k) / k_norm**2)
    dM_prev = du[:,np.newaxis] * beta / (k_norm * M_norm) * (k -
        M_prev * M_prev.dot(k.T) / M_norm**2)
    return dwt, dgamma, dwg, ds, dwc, dw_prev, dg, du, dbeta, dk, dM_prev

def forward_back_w(w_prev, M_prev, k, beta, g, s, gamma, grad_in):
    u, wc, wg, wt, w = compute_w(w_prev, M_prev, k, beta, g, s, gamma)
    return backprop_w(w_prev, M_prev, k, beta, g, s, gamma, u, wc, wg, wt,
        grad_in)

def check_head():
    n = 9
    m = 11
    w_prev = softmax(np.random.randn(n))
    M_prev = np.random.randn(n, m)
    k = np.random.randn(m)
    beta = np.random.randn()
    g = 1/(1+np.exp(-np.random.randn()))
    s = softmax(np.random.randn(n))
    gamma = np.random.randn()
    f = lambda: compute_w(w_prev, M_prev, k, beta, g, s, gamma)[-1]
    grad = forward_back_w(w_prev, M_prev, k, beta, g, s, gamma, 1)

    # check dgamma
    f_gamma = lambda gamma: compute_w(w_prev, M_prev, k, beta, g, s, gamma)[-1]
    n_dgamma = numerical_gradient_scalar(f_gamma, gamma)
    print(grad[1] - n_dgamma)

    # check ds
    n_ds = numerical_gradient_array(f, s)
    print(grad[3] - n_ds)

if __name__ == "__main__":
    check_w_last2()
