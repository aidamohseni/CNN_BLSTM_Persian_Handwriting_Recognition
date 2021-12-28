import numpy as np

x_train = np.array([[1, 1, 1, 1, 1, 1], [1, 2, 2, 2, 3, 4], [1, 1, 2, 3, 3, 3]])
y_train = np.array([[1, -1, -1, 1, 1, -1]])

# x_test = np.array([[1, 1, 1], [0, 1, 2], [0, 1, 2]])
x_test = np.array([[1, 1, 1], [0, 1, 2], [0, 1, 2]])
y_test = np.array([[1, -1, -1]])

x_train_transpose = x_train.transpose()
# print(f'Transposed Array:\n{x_train_transpose  }')
m3 = np. dot(x_train, x_train_transpose)
flipped_arr = np.fliplr(m3)
# print(f'Flliped Array:\n{flipped_arr }')
m4 = np. dot(x_train, y_train.transpose())
w = np.dot(flipped_arr, m4)
print(f'W:\n{w }')
######################
# (D-xt*w)T*(D-xt*w)
#x_test_transpose = x_test.transpose()
#o1 = np.dot(x_test_transpose, w)
#o = np.subtract(y_test, o1)
#o_transpose = o.transpose()
#mse = (np.dot(o_transpose, o))/3
#print(f'mse:\n{mse }')
######################
# 1/p [(D-y)T(D-Y)]
# (X*W)+b
######################
sum = 0
for i in range(3):
    for j in range(2):
        sum1 = x_test[i][j] * w[j+1] + w[j]
    sum += sum1
mse = sum / 3
print(f'mse:\n{mse }')
