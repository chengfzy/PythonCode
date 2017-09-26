# Ref
# [1] Wiki: https://en.wikipedia.org/wiki/Bilinear_interpolation

x = [14, 15]
y = [20, 21]
x_ = 14.5
y_ = 20.2
f = [[91, 162], [210, 95]]


# normal bilinear interpolation: https://en.wikipedia.org/wiki/Bilinear_interpolation#Algorithm
k1 = (x[1] - x_) / (x[1] - x[0]) * f[0][0] + (x_ - x[0]) / (x[1] - x[0]) * f[1][0]
k2 = (x[1] - x_) / (x[1] - x[0]) * f[0][1] + (x_ - x[0]) / (x[1] - x[0]) * f[1][1]
f0 = (y[1] - y_) / (y[1] - y[0]) * k1 + (y_ - y[0]) / (y[1] - y[0]) * k2
print("normal bilinear interpolation: {0}".format(f0))

# alternative method: https://en.wikipedia.org/wiki/Bilinear_interpolation#Nonlinear
a00 = f[0][0]
a10 = f[1][0] - f[0][0]
a01 = f[0][1] - f[0][0]
a11 = f[1][1] + f[0][0] - f[1][0] - f[0][1]
f1 = a00 + a10 * (x_ - x[0]) + a01 * (y_ - y[0]) + a11 * (x_ - x[0]) * (y_ - y[0])
print("alternative method: {0}".format(f1))
