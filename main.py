
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

np.random.seed(0)

mean = [10, 70]

# covariance matrix
cov = [[1, .95], [.95, 2]]

data = np.random.multivariate_normal(mean, cov, size=100)

df = pd.DataFrame(data, columns=["Hours Studied", "Test Score"])
df["Standardized HS"] = (df["Hours Studied"] - df["Hours Studied"].mean())/df["Hours Studied"].std()
df["Standardized TS"] = (df["Test Score"] - df["Test Score"].mean())/df["Test Score"].std()
 # standardize to make gradient descent more stable
# print(df)
# print(df.corr())
# plt.scatter(df["Hours Studied"], df["Test Score"])
# plt.show()

def mean_squared_error(m, b, points):  #  this is to calculate loss function
    total_error = 0

    for i in range(len(points)):
        x = points.iloc[i]["Standardized HS"]
        y = points.iloc[i]["Standardized TS"]
        total_error += (y - (m * x + b)) **2
    return total_error / float(len(points))

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)  # number of points

    for i in range(n):
        x = points.iloc[i]["Standardized HS"]
        y = points.iloc[i]["Standardized TS"]

        m_gradient += -(2/n) * x * (y-(m_now * x + b_now))    # these are the partial derivatives (peak ascending direction)
        b_gradient += -(2/n) * (y-(m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

m = 0
b = 0
L = 0.0025 # learning rate
epochs = 1000 # training iterations

for i in range(epochs):
    m, b = gradient_descent(m, b, df, L)
    if i % 50 ==0:
        print("Epoch", i, "MSE:", mean_squared_error(m, b, df))
print(m, b)

x_vals = np.linspace(df["Standardized HS"].min(), df["Standardized HS"].max(), 100)
y_vals = m * x_vals + b

plt.scatter(df["Standardized HS"], df["Standardized TS"])
plt.plot(x_vals, y_vals, color = "red")
plt.show()


# now we can unstandardize

mean_x = df["Hours Studied"].mean()
std_x = df["Hours Studied"].std()
mean_y = df["Test Score"].mean()
std_y = df["Test Score"].std()

M = m * (std_y / std_x)
B = -m * (std_y / std_x) * mean_x + b * std_y + mean_y

x_vals = np.linspace(df["Hours Studied"].min(), df["Hours Studied"].max(), 100)
y_vals = M * x_vals + B

plt.scatter(df["Hours Studied"], df["Test Score"])
plt.plot(x_vals, y_vals, color = "red")
plt.show()
