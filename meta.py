#meta.py

import copy

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.optim import Optimizer

import sys
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.nonlin = torch.tanh

        self.lin1 = torch.nn.Linear(1, 1)
        #self.lin2 = torch.nn.Linear(1, 1)



    def forward(self, x):

        #x = x * 0

        #return self.lin1(x) + self.lin2(x - 0.5)
        return self.lin1(x)


def calculateGradVector(model, sizeTotal):

    count1 = 0
    gradNow = np.zeros((sizeTotal,))
    for param in model.parameters():
        size1 = param.nelement()
        grad1 = param.grad
        #print (grad1)
        grad1 = grad1.data.numpy()
        grad1 = grad1.reshape((size1,))

        gradNow[count1:count1+size1] = np.copy(grad1)
        count1 += size1

    return gradNow


def findDataGrad(X, Y, model, optimizer, sizeTotal):

    dataGrad = np.zeros((X.shape[0],  sizeTotal))

    for a in range(X.shape[0]):
        pred1 = model(X[a:a+1])
        loss1 = (pred1[0] - Y[a]) ** 2
        loss1.backward()

        gradNow = calculateGradVector(model, sizeTotal)
        dataGrad[a] = np.copy(gradNow)

        optimizer.zero_grad()

    return dataGrad

def getOptSquare(optimizer, sizeTotal):

    count1 = 0
    stateVal = np.zeros(sizeTotal)
    for state1 in optimizer.state.values():
        param = state1['square_avg']
        size1 = param.nelement()

        param = param.data.numpy()

        stateVal[count1:count1+size1] = np.copy(param)
        count1 += size1

    return stateVal


def trainModel():

    N = 100

    #X = np.random.random(N)
    X = np.arange(N).astype(float) / N



    #Y = np.exp(X*3) / np.exp(3)
    Y = np.copy(X)
    Y[Y>0.9] = Y[Y>0.9] + ((X[Y>0.9] - 0.9) * 3)



    X = X.reshape((-1, 1))

    #X = X - np.mean(X)
    #X = X - 1
    X = X - 0.5
    #X = X - 0.945
    #X = X - 0.95

    X = torch.tensor(X).float()
    Y = torch.tensor(Y).float()

    weight1 = torch.zeros(Y.shape[0]).float()



    model = SimpleModel()
    # model = torch.load('./models/3.pt')

    sizeTotal = 0
    for param in model.parameters():
        size1 = param.nelement()
        sizeTotal += size1


    argKeep = np.arange(N)

    Nreq = 5#20
    dataVlues = np.zeros((Nreq, N)) - 1

    nPrint = 100

    #learningRate = 1e-1
    #learningRate = 5e-2
    #learningRate = 2e-2
    learningRate = 1e-2
    #learningRate = 1e-4
    #optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)
    optimizer = torch.optim.RMSprop(model.parameters(), lr = learningRate, alpha=0.95)
    #optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)
    #optimizer = RMSprop(model.parameters(), lr = learningRate)

    iterNum = 10000

    paramList = np.zeros((iterNum, 2))
    keepSize = np.zeros(iterNum)

    validGradSum = np.zeros(sizeTotal)
    validAbsSum = np.zeros(sizeTotal)


    for iter in range(0, iterNum):

        dataGrad = findDataGrad(X, Y, model, optimizer, sizeTotal)

        if iter > 50:

            stateVal = getOptSquare(optimizer, sizeTotal)
            stateVal = stateVal ** 0.5

            if iter % 100 == 0:
                print ('stateVal', stateVal[1], stateVal[0])

            stateVal = stateVal.reshape((1, stateVal.shape[0]))
            dataGrad = dataGrad * (1 / stateVal)



        #'''
        #argsort1 = np.argsort(pred.data.numpy())
        argsort1 = np.argsort(Y.data.numpy())
        topX = argsort1[  -argsort1.shape[0] // 10:  ]

        pred = model(X[topX])
        pred = pred[:, 0]

        #print (pred.shape)
        #print (Y[topX].shape)


        #validLoss = torch.mean( (pred - Y[topX] - (X[topX, 0] * 50)  ) ** 2  )
        validLoss = torch.mean( (pred - Y[topX]) ** 2  )
        validLoss.backward()
        #'''

        #pred = model(X[-10:])
        #pred = pred[:, 0]
        #loss = torch.mean( (pred - Y[-10:]  ) ** 2  )
        #loss.backward()

        #plt.plot(X[topX, 0].data.numpy(), Y[topX].data.numpy() + (X[topX, 0] * 50).data.numpy() )
        #plt.plot(X[topX, 0].data.numpy(), pred.data.numpy()  )
        #plt.show()


        #for param in model.parameters():
        #    grad1 = param.grad
        #    print (grad1)
        #quit()

        beta1 = 0.02
        validGrad = calculateGradVector(model, sizeTotal)
        validGradSum = (validGradSum * (1 - beta1)) + (validGrad * beta1)
        validAbsSum = (validAbsSum * (1 - beta1)) + (np.abs(validGrad) * beta1)

        #print (validGrad[1] * -1, validGrad[0] * -1)
        #quit()
        validDir = validGradSum / validAbsSum
        validDir = validDir.reshape((1, validDir.shape[0]))
        dataValue = np.sum(validDir * dataGrad, axis=1)

        dataVlues[iter % Nreq] = np.copy(dataValue)

        #pred = model(X[argKeep])
        #pred = pred[:, 0]
        #loss = torch.mean( (pred - Y[argKeep]) ** 2  )

        pred = model(X)
        pred = pred[:, 0]
        loss = torch.mean( weight1 * (pred - Y) ** 2  )


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #for param_group in optimizer.param_groups:
        #    print (param_group)
        #    print(param_group['lr'])

        #print ([optimizer.state])
        #print ('')
        #print ([optimizer.state.keys()])
        #print (optimizer.state.values())



        #print ("")
        a0 = 0
        for param in model.parameters():
            paramList[iter, a0] = param[0]
            a0 += 1

        if iter%nPrint == 0:
            #plt.plot(paramList[iter-100:iter, 1], paramList[iter-100:iter, 0])
            #plt.scatter(paramList[iter-100:iter, 1], paramList[iter-100:iter, 0])
            #plt.show()

            plt.plot(paramList[iter-100:iter, 1])
            plt.plot(paramList[iter-100:iter, 0])
            plt.plot(keepSize[iter-100:iter] / 100)
            plt.show()


        if iter % nPrint == 0:

            #Slope, Intercept, loss grad
            print ('')
            print ('Param', paramList[iter, 1], paramList[iter, 0])
            print ('validGrad', validDir[0][1] * -1, validDir[0][0] * -1)
            #print (loss)
            print ('Mean', np.mean(pred[topX].data.numpy()), np.mean(Y[topX].data.numpy()) )

            dataValue_plot = dataValue / np.mean(np.abs(dataValue))
            dataGrad_plot = -1 * dataGrad / np.mean(np.abs(dataGrad))

            #plt.scatter(X[:, 0].data.numpy(), Y.data.numpy())
            #print (pred.shape, argKeep.shape)
            #plt.scatter(X[argKeep, 0].data.numpy(), pred.data.numpy())
            #plt.scatter(X[:, 0].data.numpy(), dataValue_plot)
            #plt.scatter(X[topX, 0].data.numpy(), Y[topX].data.numpy())
            #plt.scatter(X[:, 0].data.numpy(), dataGrad[:, 0]*-1)
            #plt.scatter(X[:, 0].data.numpy(), dataGrad[:, 1]*-1)
            #plt.show()

            figure, axis = plt.subplots(2)
            axis[0].plot(X[:, 0].data.numpy(), Y.data.numpy())
            axis[0].scatter(X[argKeep, 0].data.numpy(), pred.data.numpy())
            axis[0].plot(X[topX, 0].data.numpy(), Y[topX].data.numpy())

            axis[1].scatter(X[:, 0].data.numpy(), dataGrad_plot[:, 1])
            axis[1].scatter(X[:, 0].data.numpy(), dataGrad_plot[:, 0])
            axis[1].plot(X[:, 0].data.numpy(), dataValue_plot)

            axis[1].plot(X[:, 0].data.numpy(), weight1.data.numpy())
            plt.show()


        weight1[dataValue > 0] = weight1[dataValue > 0] + 0.05
        weight1[dataValue < 0] = weight1[dataValue < 0] - 0.05
        weight1[weight1 < 0] = 0
        weight1[weight1 > 1] = 1



        #if iter % nPrint == 0:
        #argKeep1 = np.argwhere(dataValue > 0)[:, 0]
        #argKeep2 = np.argsort(dataValue)[  -dataValue.shape[0] // 10:  ]
        #argKeep = np.concatenate((argKeep1, topX))
        #argKeep = np.unique(argKeep)

        #maxVal = np.max(dataVlues, axis=0)
        #argKeep = np.argwhere(maxVal > 0)[:, 0]

        #keepSize[iter] = argKeep.shape[0]

        #torch.save(model, './models/3.pt')








trainModel()




#
