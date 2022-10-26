from pickle import NONE
import numpy as np
import matplotlib.pyplot as plt

#生成資料
def generate_linear(n=100):
    """
    Generate data points which are linearly separable
    :param n: number of points
    :return: inputs and labels
    """
    pts = np.random.uniform(0, 1, (n, 2))
    inputs, labels = [], []

    for pt in pts:
        inputs.append([pt[0], pt[1]])
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)

    return np.array(inputs), np.array(labels).reshape(n, 1)



def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i,0.1*i])
        labels.append(0)
        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i,1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21,1)



def show_result(x, y, y_pred):
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize = 18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0],x[i][1], 'ro')
        else:
            plt.plot(x[i][0],x[i][1], 'bo')

    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize = 18)
    for i in range(x.shape[0]):
        if y_pred[i] == 0:
            plt.plot(x[i][0],x[i][1], 'ro')
        else:
            plt.plot(x[i][0],x[i][1], 'bo')

    plt.show()

x1, y1 = generate_linear(n=100)
x2, y2 = generate_XOR_easy()


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def d_sigmoid(x):
    return np.multiply(x, 1.0 - x)


def relu(x):
    return np.multiply(x, (x > 0))


def d_relu(x):
    return np.multiply(1., (x > 0))


class network:
    def __init__(self, neu_num1, neu_num2):
        self.w0 = np.random.normal(0, 1, size = (2, neu_num1))
        self.w0 *= 0.1
        self.w1 = np.random.normal(0, 1, size = (neu_num1, neu_num2))
        self.w1 *= 0.1
        self.w2 = np.random.normal(0, 1, size = (neu_num2, 1))
        self.w2 *= 0.1


    def losswithoutact(self, groundtruth):
            return np.mean((groundtruth - self.o2)**2)/2


    def forwardwithoutact(self, x):
        self.x = x
        self.o0 = np.dot(self.x, self.w0)
        self.o1 = np.dot(self.o0, self.w1)
        self.o2 = np.dot(self.o1, self.w2)
        self.y = self.o2

        return self.y


    def backwardwithoutact(self, groundtruth):
        dLdy = (groundtruth - self.y)
        self.dLdw2 = np.dot(self.o1.T, dLdy)


        dLdydo1dw1 = np.dot(dLdy, self.w2.T)
        self.dLdw1 = np.dot(self.o0.T, dLdydo1dw1)


        dLdydo1do0dw0 = np.dot(dLdydo1dw1, self.w1.T)
        self.dLdw0 = np.dot(self.x.T, dLdydo1do0dw0)
    

    def optimizer(self, lr = 0.01):
        self.w2 += lr * self.dLdw2
        self.w1 += lr * self.dLdw1
        self.w0 += lr * self.dLdw0


    def trainwithoutact(self, x, groundtruth, epoch = 1000, lr = 0.1):
        self.losslist = []
        for i in range(epoch):
            self.forwardwithoutact(x)
            self.backwardwithoutact(groundtruth)
            self.optimizer(lr)
            if (i%10000 == 0):
                print("epoch: " + str(i) + "    Loss: " + str(np.mean((groundtruth - self.y)**2)))
            self.losslist.append(np.mean((groundtruth - self.y)**2))
        print("predction: \n", self.y)
        plt.plot(np.arange(epoch), self.losslist)
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.show()
        self.show_accuracywithoutact(groundtruth)


    def show_accuracywithoutact(self, groundtruth):
        self.pred_y = self.y
        acc = 0
        for i in range(len(self.pred_y)):
            if self.y[i][0] >= 0.5:
                self.pred_y[i][0] = 1
            else:
                self.pred_y[i][0] = 0
        for j in range(len(self.pred_y)):
            if self.pred_y[j][0] == groundtruth[j][0]:
                acc+=1
        acc = (acc/len(self.pred_y))*100
        print(" accuracy = " + str(acc) + "%")
        show_result(self.x, groundtruth, self.pred_y)


a = network(5, 5)
a.trainwithoutact(x1, y1, 10000, 0.01)
