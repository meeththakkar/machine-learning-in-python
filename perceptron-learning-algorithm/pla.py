import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl

class PLA(object):
    """PLA algorithm class
    This class implements Perceptron learning algorithm.d
    """

    def __init__(self):
        self.w = np.zeros(3)
        self.w[0] = 0
        self.w[1] = 0
        self.w[2] = 0
        self.learningRate = 0.05

    def response(self,x):
        resp =   self.w[0]*x[0] + self.w[1]*x[1] + self.w[2]*x[2]
        if resp > 0:
            lable = 1
        else:
            lable = -1

        return lable

    def run(self,input):
        globalError = 0
        learning = True
        iteration_number = 0
        Correctlabel = 0
        while learning:
            for x in input:
                resp = self.response(x)
                if resp != x[3]:
                    Correctlabel = x[3]
                    self.w[0] += self.learningRate * Correctlabel * x[0]
                    self.w[1] += self.learningRate * Correctlabel * x[1]
                    self.w[2] += self.learningRate * Correctlabel * x[2]
                    globalError += Correctlabel
            iteration_number += 1
            #print('iterations:', iteration_number)
            if globalError == 0.0 or iteration_number >= 1000:  # stop criteria
                #print ('iterations:', iteration_number)
                learning = False  # stop learing
                print("final global error", globalError)
                print("weigh (w) vector",self.w)

def generateData(n, slope, offset, distance, xmax, ymax):
    """Generate data for PLA algorithm.
    n - sample size
    slope = slope of separating line
    offset = offset of separating line
    distance = minimum distance between two clusters.
    xmax = max xdimension
    ymax - maxy dimension
    :rtype: numpyarray
    """
    inputs = np.zeros([n,4],dtype=float)
    for i in range(n):
        x = np.random.rand() * xmax
        y = np.random.rand() * ymax + offset
        if (y > (slope * x + offset)):
            label = 1
            y = y + distance
        else:
            label = -1
            y = y - distance
        inputs[i,0] = 1
        inputs[i,1] = x
        inputs[i,2] = y
        inputs[i,3] = label
    return inputs

def main():
    print("start PLA")
    data = generateData(500, 0.1, -5 , 0.3, 10, 10)
    print(data)
    plt.plot(data[:,1],data[:,2],'ob')
    pyl.show()
    pla =  PLA()
    pla.run(data)
    print("Testing perceptron")
    print("response expected -1 " , pla.response([1,1,0.8]))
    print("response expected  1 " , pla.response([1,1,1.8]))
    print("response expected -1 " , pla.response([1,10,0.8]))
    print("response expected  1 " , pla.response([1,10,800]))
    print("response expected  1 " , pla.response([1,0.1,0.8]))
    print("response expected  1 " , pla.response([1,0.0001,0.1]))


if __name__ == "__main__":
    main()
