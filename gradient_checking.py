import numpy as np

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
def numerical_gradient_scalar(f, param):
    param += epsilon
    f1 = f()
    param -= 2*epsilon
    f2 = f()
    param += epsilon
    return (f1-f2)/(2*epsilon)

def check_numerical_gradient():
    a = np.array([1.0, 2.0])
    f = lambda: a**2
    n_grad = numerical_gradient_array(f, a)
    grad = 2*a
    diff = n_grad-grad
    print(diff)

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
    f = lambda: w**gamma / (w**gamma).sum()
    n_grad = numerical_gradient_scalar(f, gamma)
    print(n_grad)

if __name__ == "__main__":
    check_w_tilde()
