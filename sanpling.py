import numpy as np
from scipy.stats import truncnorm
from scipy.stats import norm

delta = 0.05

X1 = np.random.uniform(0, 0.2, 4000)
X2 = np.random.uniform(0.2, 0.4, 4000)
X3 = np.random.uniform(0.4, 0.6, 4000)
X4 = np.random.uniform(0.6, 0.8, 4000)
X5 = np.random.uniform(0.8, 1.0, 4000)
X = np.hstack([X1, X2, X3, X4, X5])

y1 = np.random.binomial(1, 0.1, 4000)
y2 = np.random.binomial(1, 0.3, 4000)
y3 = np.random.binomial(1, 0.5, 4000)
y4 = np.random.binomial(1, 0.7, 4000)
y5 = np.random.binomial(1, 0.9, 4000)
y = np.hstack([y1, y2, y3, y4, y5])

z1 = np.vstack([X1, y1]).T
z2 = np.vstack([X2, y2]).T
z3 = np.vstack([X3, y3]).T
z4 = np.vstack([X4, y4]).T
z5 = np.vstack([X5, y5]).T

mean = 0.5
std = 0.4
a = 0.0
b = 1.0

# a, b = (a - mean) / std, (b - mean) / std
# cdf1 = truncnorm.cdf(0.0, a, b, loc=mean, scale=std)
# cdf2 = truncnorm.cdf(0.2, a, b, loc=mean, scale=std)
# cdf3 = truncnorm.cdf(0.4, a, b, loc=mean, scale=std)
# cdf4 = truncnorm.cdf(0.6, a, b, loc=mean, scale=std)
# cdf5 = truncnorm.cdf(0.8, a, b, loc=mean, scale=std)
# cdf6 = truncnorm.cdf(1.0, a, b, loc=mean, scale=std)
cdf1 = norm.cdf(0.0, loc=mean, scale=std)
cdf2 = norm.cdf(0.2, loc=mean, scale=std)
cdf3 = norm.cdf(0.4, loc=mean, scale=std)
cdf4 = norm.cdf(0.6, loc=mean, scale=std)
cdf5 = norm.cdf(0.8, loc=mean, scale=std)
cdf6 = norm.cdf(1.0, loc=mean, scale=std)
b1 = cdf2 - cdf1
b2 = cdf3 - cdf2
b3 = cdf4 - cdf3
b4 = cdf5 - cdf4
b5 = cdf6 - cdf5

weights = norm.pdf(X, loc=mean, scale=std) / 1.0
estimated_safety = np.mean(weights * y)
true_safety = b1 * 0.1 + b2 * 0.3 + b3 * 0.5 + b4 * 0.7 + b5 * 0.9
M = 1 / np.sqrt(2 * np.pi) / std
slack = M * np.sqrt(2/delta/X.shape[0])
safety_range_low = estimated_safety - slack
safety_range_high = estimated_safety + slack

print('True safety', true_safety)
print("Estimated safety", estimated_safety)
print('Estimated lower bound', safety_range_low)
print('Estimated upper bound', safety_range_high)