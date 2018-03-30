import numpy as np
from loader import MNIST
from nnCostFunction import nnCostFunction
from randInitializeWeights import randInitializeWeights
from computeNumericalGradient import unRolling
from predict import predict
from shuffle import shuffle
# Get data from Mnist
data = MNIST()
data.load_training()
data.load_testing()

x_train = data.train_images
y_train = data.train_labels

x_test = data.test_images
y_test = data.test_labels

x_train = np.reshape(x_train, (len(x_train), 784))
y_train = np.reshape(y_train, (len(y_train), 1))
y_train_fix = np.reshape(np.zeros(len(y_train)*10), (len(y_train), 10))

for i in range(len(y_train)):
    for j in range(0, 10):
        if y_train[i] == j:
            y_train_fix[i][j] = 1

# Create Validation, Train
list_x_val = []
list_y_val = []
list_x_train = []
list_y_train = []
list_y_val_fix = []
for i in range(6):
    x_val = x_train[i*10000: i*10000 + 10000]
    y_val = y_train[i*10000: i*10000 + 10000]
    y_val_fix = y_train_fix[i*10000: i*10000 + 10000]

    x_traint = np.concatenate(
        [x_train[0:i*10000], x_train[i*10000 + 10000:60000]])
    y_traint = np.concatenate(
        [y_train_fix[0:i*10000], y_train_fix[i*10000 + 10000:60000]])

    list_x_val.append(x_val)
    list_y_val.append(y_val)
    list_y_val_fix.append(y_val_fix)

    list_x_train.append(x_traint)
    list_y_train.append(y_traint)

# initialize theta
hidden_layer = 150

# mini batch
x = np.zeros((1000, 784))
y = np.zeros((1000, 10))

# train process
alpha = 0.01
epoch = 148
mini_batch = 80
l = 50 #lamda
index = 5
for index in range(6):
    x_train = list_x_train[index]
    y_train = list_y_train[index]
    theta1 = randInitializeWeights(784, hidden_layer)
    theta2 = randInitializeWeights(hidden_layer, hidden_layer)
    theta3 = randInitializeWeights(hidden_layer, 10)
    print 'training... ' + str(index + 1)
    for i in range(epoch):
        # mini batch
        x_train, y_train = shuffle(x_train, y_train)
        for k in range(len(x_train)/mini_batch):
            x = x_train[k*mini_batch:(k + 1)*mini_batch]
            y = y_train[k*mini_batch:(k + 1)*mini_batch]

            results = nnCostFunction(theta1, theta2, theta3, x, y, lamda=l)
            grad1 = results[1]
            grad2 = results[2]
            grad3 = results[3]
            #print 'inter ' + str(i) + ' coss = ' + str(results[0])
            grad = np.concatenate([unRolling(grad1), unRolling(grad2)])
            grad = np.concatenate([grad, unRolling(grad3)])
            theta = np.concatenate([unRolling(theta1), unRolling(theta2)])
            theta = np.concatenate([theta, unRolling(theta3)])
            theta = theta - alpha * grad
            theta1 = np.reshape(
                theta[0:785 * hidden_layer], (hidden_layer, 785))
            theta2 = np.reshape(
                theta[785 * hidden_layer:(786 + hidden_layer) * hidden_layer], (hidden_layer, hidden_layer + 1))
            theta3 = np.reshape(
                theta[(786 + hidden_layer) * hidden_layer:len(theta)], (10, hidden_layer + 1))
        j_train = nnCostFunction(theta1, theta2, theta3,
                             x_train, y_train, lamda=l)[0]
        # print 'epoch ' + str(i + 1) + ' coss train = ' + str(j_train)

    j_val = nnCostFunction(
        theta1, theta2, theta3, list_x_val[index], list_y_val_fix[index], lamda=l)[0]
    j_train = nnCostFunction(theta1, theta2, theta3,
                             x_train, y_train, lamda=l)[0]
    print 'coss val = ' + str(j_val) + ' coss train = ' + str(j_train) + ' lamda = ' + str(l)
    theta = np.concatenate([unRolling(theta1), unRolling(theta2)])
    theta = np.concatenate([theta, unRolling(theta3)])
    np.savetxt('theta-val-version 4- {}.txt'.format(index), theta, delimiter=',')
    np.savetxt('theta-val-version 4- {}.txt'.format(index), j_val, delimiter=',')
    np.savetxt('theta-val-version 4- {}.txt'.format(index), j_train, delimiter=',')
    a4 = predict(theta1, theta2, theta3, list_x_val[index])
    count = 0
    for i in range(len(a4)):
        a4[i] = np.argmax(a4[i])
        if a4[i][0] == list_y_val[index][i]:
            count += 1

    print 'val - ' + str(index) + ' ' + str((100.0 * count/10000))
