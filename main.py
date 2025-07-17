
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)

mean = [10, 70]

# covariance matrix
cov = [[1, .95], [.95, 2]]

data = np.random.multivariate_normal(mean, cov, size=100)

df = pd.DataFrame(data, columns=["Hours Studied", "Test Score"])

# print(df)
# print(df.corr())
# plt.scatter(df["Hours Studied"], df["Test Score"])
# plt.show()

def mean_squared_error(m, b, points):  #  this is to calculate loss function
    total_error = 0

    for i in range(len(points)):
        x = points.iloc[i]["Hours Studied"]
        y = points.iloc[i]["Test Score"]
        total_error += (y - (m * x + b)) **2
    total_error /= float(len(points))

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)  # number of points

    for i in range(n):
        x = points.iloc[i]["Hours Studied"]
        y = points.iloc[i]["Test Score"]

        m_gradient += -(2/n) * x * (y-(m_now * x + b_now))    # these are the partial derivatives (peak ascending direction)
        b_gradient += -(2/n) * (y-(m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

m = 0
b = 0
L = 0.001 # learning rate
epochs = 1000 # training iterations

for i in range(epochs):
    m, b = gradient_descent(m, b, df, L)
    if i % 50 ==0:
        print(i)
print(m, b)

plt.scatter(df["Hours Studied"], df["Test Score"])
plt.plot(list(range(6, 12)), [ m * x + b for x in range(6, 12)], color = "red")
plt.show()
