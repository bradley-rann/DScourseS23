import random
import numpy as np
from scipy.optimize import minimize
import statsmodels.api as sm



#Create data set
random.seed(100)
N = 100000
K = 10
X = np.random.normal(size=(N, K-1))
X = np.insert(X, 0, 1, axis=1)

eps = np.random.uniform(low=0, high=0.25, size=N)
beta = np.array([1.5, -1, -0.25, 0.75, 3.5, -2, 0.5, 1, 1.25, 2])
Y = X.dot(beta)+eps

#OLS estimates

beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
print(beta_hat)

#gradient estimate

def gradient(X, Y, beta):
    return 2 * X.T.dot(X.dot(beta) - Y) / len(Y)

def gradient_descent(X, Y, alpha, iterations):
    beta = np.zeros(X.shape[1])
    for i in range(iterations):
        beta = beta - alpha * gradient(X, Y, beta)
    return beta

beta_hat_gradient_descent = gradient_descent(X, Y, alpha=0.0000003, iterations=20000)
print(beta_hat_gradient_descent)

#Using BFGS

def objective(beta, X, Y):
    return np.sum((X.dot(beta) - Y) ** 2)

beta_init = np.zeros(X.shape[1])
result = minimize(objective, beta_init, args=(X, Y), method='L-BFGS-B')

beta_hat_OLS = result.x
print(beta_hat_OLS)

#Using Nelder-Mead
# Objective Function def
def objective(beta, X, Y):
    return np.sum((X.dot(beta) - Y)**2)

# Initialize beta
beta0 = np.zeros(X.shape[1])

# Nelder-Mead
res = minimize(objective, beta0, args=(X, Y), method='Nelder-Mead')

# Print output
print(res.x)


#Beta_hat MLE

#Likelihood
def neg_log_likelihood(beta, X, Y):
    N = len(Y)
    sigma = 0.25
    error = Y - X.dot(beta)
    return N/2*np.log(2*np.pi*sigma**2) + 1/(2*sigma**2)*np.sum(error**2)

#Gradient
def grad_neg_log_likelihood(beta, X, Y):
    sigma = 0.25
    error = Y - X.dot(beta)
    return -1/(sigma**2) * X.T.dot(error)

beta0 = np.zeros(K)
out = minimize(neg_log_likelihood, beta0, method='L-BFGS-B', args=(X, Y), jac=grad_neg_log_likelihood)

print(out.x)


#Easy way

model = sm.OLS(Y,X).fit()
print(model.summary())


with open('C:/Users/radle/Downloads/regression_table.tex', 'w') as f:
    f.write(model.summary().as_latex())


easy_beta_hat = model.params
print(easy_beta_hat)
