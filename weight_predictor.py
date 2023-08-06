#meta.py


from matplotlib import image
from matplotlib import pyplot
import matplotlib.pyplot as plt

import numpy as np
import os


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.optim import Optimizer

import scipy
from scipy import stats
import copy


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        #channel2 = 16

        #channel1 = 16
        channel1 = 32
        channel2 = 16
        maxPool2 = 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=channel1,
                kernel_size=7,
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        #self.conv2 = nn.Sequential(
        #    nn.Conv2d(channel1, channel2, 7, 1),
        #    nn.ReLU(),
        #    nn.MaxPool2d(maxPool2),
        #)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel1, channel2, 7, 1),
            nn.Tanh(),
            nn.MaxPool2d(maxPool2),
        )
        # fully connected layer, output 10 classes
        #self.out = nn.Linear(32 * 7 * 7, 10)
        #self.out = nn.Linear(channel2 * 8 * 8, 1)
        #self.out = nn.Linear(channel2 * 7 * 7, 1)
        self.out = nn.Linear(channel2 * 5 * 5, 1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x    # return x for visualization




class returnVector(nn.Module):
    def __init__(self, N):
        super(returnVector, self).__init__()

        channel2 = 16

        self.Weights = torch.nn.Parameter(torch.rand(N) - 0.5)


    def forward(self):

        return self.Weights


class linearOutput(nn.Module):
    def __init__(self):
        super(linearOutput, self).__init__()

        channel2 = 16

        #self.out = nn.Linear( (channel2 * 5 * 5) + 3 , 1)
        self.out = nn.Linear( 8 , 1)


    def forward(self, x):

        return self.out(x), ''


class weightModel(nn.Module):
    def __init__(self):
        super(weightModel, self).__init__()

        channel2 = 16

        #self.lin1 = nn.Linear( (channel2 * 5 * 5) + 1 , 3)
        self.lin1 = nn.Linear( 2 , 20)

        self.lin2 = nn.Linear( 20  , 1)


    def forward(self, x):

        x = self.lin1(x)
        x = torch.tanh(x)
        x = self.lin2(x)

        return x, ''


class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()

        #channel2 = 16

        channel1 = 16
        channel2 = 16
        maxPool2 = 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=channel1,
                kernel_size=7,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel1, channel2, 7, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(maxPool2),
        )
        # fully connected layer, output 10 classes
        #self.out = nn.Linear(32 * 7 * 7, 10)
        self.out = nn.Linear(channel2 * 8 * 8, 1)
        #self.out = nn.Linear(channel2 * 5 * 5, 1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x    # return x for visualization


def loadnpz(name, allow_pickle=False):
    if allow_pickle:
        data = np.load(name, allow_pickle=True)
    else:
        data = np.load(name)
    data = data.f.arr_0
    return data



def doDownSample(image, downSize):

    size1 = image.shape[0] // downSize
    image = image[:(size1*downSize), :(size1*downSize)]

    image = image.reshape((image.shape[0] // downSize, downSize, image.shape[1] // downSize, downSize, 3))
    image = np.sum(image, axis=(1, 3))
    image = image // (downSize **  2)

    return image


def downSampler():

    folder1 = './Age/crop_part1/'
    fNames = os.listdir(folder1)

    downSize = 5

    allImages = np.zeros((len(fNames), 200 // downSize, 200 // downSize, 3), dtype=int)
    ageList = np.zeros(len(fNames), dtype=int)

    for a in range(len(fNames)):

        print (a)
        fname = fNames[a]

        age1 = int(fname.split('_')[0])

        ageList[a] = age1


        image2 = image.imread(folder1 + fname)


        image2 = doDownSample(image2, downSize)

        allImages[a] = np.copy(image2)

    np.savez_compressed('./Age/data/allImages.npz', allImages)
    np.savez_compressed('./Age/data/allAges.npz', ageList)



#downSampler()
#quit()



def original_trainModel():


    X = loadnpz('./Age/data/allImages.npz')
    Y = loadnpz('./Age/data/allAges.npz')

    Y = Y / 100
    X = np.swapaxes(X, 1, 3)
    X = np.swapaxes(X, 2, 3)


    N = Y.shape[0]
    perm1 = np.random.permutation(N)
    X = X[perm1]
    Y = Y[perm1]
    X = torch.tensor(X).float()
    Y = torch.tensor(Y).float()

    N_test = 1000
    X_test = X[:N_test]
    Y_test = Y[:N_test]
    X = X[N_test:]
    Y = Y[N_test:]

    N_train = Y.shape[0]

    #print (X.shape)
    #quit()



    model = CNN()

    #learningRate = 1e-3
    learningRate = 5e-4
    #learningRate = 2e-4
    optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)


    iterN = 100
    batchSize = 100
    Nbatch = N_train // batchSize

    for iter in range(iterN):

        for batchNum in range(Nbatch):

            #if (batchNum % 100 == 0):
            #    print ('batchnum: ', batchNum, Nbatch)

            X_batch = X[batchNum*batchSize:(batchNum+1)*batchSize]
            Y_batch = Y[batchNum*batchSize:(batchNum+1)*batchSize]

            pred, _ = model(X_batch)
            pred = pred[:, 0]

            loss = torch.mean((pred - Y_batch) ** 2)

            #print (loss)
            #cor1 = scipy.stats.pearsonr(pred.data.numpy(), Y_batch.data.numpy())

            #print (cor1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        pred_test, _ = model(X_test)
        pred_test = pred_test[:, 0]

        cor_test = scipy.stats.pearsonr(pred_test.data.numpy(), Y_test.data.numpy())

        pred_train, _ = model(X[:N_test])
        pred_train = pred_train[:, 0]

        cor_train = scipy.stats.pearsonr(pred_train.data.numpy(), Y[:N_test].data.numpy())

        #print ("Scores")
        print ('iter', iter)
        print (cor_train)
        print (cor_test)





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

def findParamVec(model, sizeTotal):

    count1 = 0
    paramVec = np.zeros((sizeTotal,))

    for param in model.parameters():
        size1 = param.nelement()
        param1 = param.data.numpy()
        param1 = param1.reshape((size1,))

        #print (param1)

        paramVec[count1:count1+size1] = np.copy(param1)
        count1 += size1
    #print (paramVec)
    #quit()
    return paramVec



def findDataGrad(X, Y, model, optimizer, sizeTotal):

    dataGrad = np.zeros((X.shape[0],  sizeTotal))

    for a in range(X.shape[0]):
        pred1, _ = model(X[a:a+1])

        loss1 = (pred1[0, 0] - Y[a]) ** 2
        loss1.backward()

        gradNow = calculateGradVector(model, sizeTotal)
        dataGrad[a] = np.copy(gradNow)

        optimizer.zero_grad()

    return dataGrad



def trainModel():

    useMeta = False


    X = loadnpz('./Age/data/allImages.npz')
    Y = loadnpz('./Age/data/allAges.npz')

    #print (X.shape)
    #quit()

    #plt.hist(Y, bins=100)
    #plt.show()
    #quit()

    Y = Y / 100
    X = np.swapaxes(X, 1, 3)
    X = np.swapaxes(X, 2, 3)

    minAge_test = 0.14
    maxAge_test = 0.21

    #minAge_valid = 0.12
    #maxAge_valid = 0.23
    minAge_valid = 0.14
    maxAge_valid = 0.21





    N = Y.shape[0]
    rng = np.random.RandomState(0)
    perm1 = rng.permutation(N)
    X = X[perm1]
    Y = Y[perm1]

    N_test = 1000
    X_test = X[:N_test]
    Y_test = Y[:N_test]
    X = X[N_test:]
    Y = Y[N_test:]
    if useMeta:
        N_valid = 1000
        X_valid = X[:N_valid]
        Y_valid = Y[:N_valid]
        X = X[N_valid:]
        Y = Y[N_valid:]


        validArg = np.argwhere(np.logical_and(Y_valid >= minAge_valid, Y_valid <= maxAge_valid))[:, 0]
        X_valid = X_valid[validArg]
        Y_valid = Y_valid[validArg]

        X_valid = torch.tensor(X_valid).float()
        Y_valid = torch.tensor(Y_valid).float()


    X = torch.tensor(X).float()
    Y = torch.tensor(Y).float()
    X_test = torch.tensor(X_test).float()
    Y_test = torch.tensor(Y_test).float()




    metaTest = False

    if metaTest:
        testArg = np.argwhere(np.logical_and(Y_test.data.numpy() >= minAge_test, Y_test.data.numpy() <= maxAge_test))[:, 0]
        X_test = X_test[testArg]
        Y_test = Y_test[testArg]


    restrictTrain = False
    if restrictTrain:
        trainArg = np.argwhere(np.logical_and(Y.data.numpy() >= minAge_test, Y.data.numpy() <= maxAge_test))[:, 0]
        X = X[trainArg]
        Y = Y[trainArg]

    #print (X.shape)
    #quit()


    N_train = Y.shape[0]


    #model = torch.load('./models/2.pt')
    model = CNN()


    sizeTotal = 0
    for param in model.parameters():
        size1 = param.nelement()
        sizeTotal += size1



    #learningRate = 1e-3
    #learningRate = 5e-4
    #learningRate = 3e-4
    learningRate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)


    iterN = 100
    batchSize = 100
    Nbatch = N_train // batchSize

    trainLoss = []

    for iter in range(iterN):

        for batchNum in range(Nbatch):

            X_batch = X[batchNum*batchSize:(batchNum+1)*batchSize]
            Y_batch = Y[batchNum*batchSize:(batchNum+1)*batchSize]

            if useMeta:
                dataGrad = findDataGrad(X_batch, Y_batch, model, optimizer, sizeTotal)


                pred_valid, _ = model(X_valid)
                pred_valid = pred_valid[:, 0]
                loss_Valid = torch.mean((pred_valid - Y_valid) ** 2)
                loss_Valid.backward()

                validGrad = calculateGradVector(model, sizeTotal)
                validDir = validGrad
                validDir = validDir.reshape((1, validDir.shape[0]))

                dataValue = np.sum(validDir * dataGrad, axis=1)

                #norm1 = np.sum(validDir**2, axis=1) ** 0.5
                #norm2 = np.sum(dataGrad**2, axis=1) ** 0.5
                #dataValue = dataValue / (norm1 * norm2)

                #plt.scatter(Y_batch.data.numpy(), dataValue)
                #plt.show()
                #quit()

                #weighting = np.zeros(dataValue.shape[0])
                #weighting[dataValue > 0] = 1

                weighting = np.copy(dataValue) / 2
                weighting[weighting<0] = 0
                weighting[weighting>1] = 1

                #print (torch.mean(Y_batch[dataValue > 0]), torch.mean(Y_batch))



                weighting = torch.tensor(weighting).float()




            pred, _ = model(X_batch)
            pred = pred[:, 0]
            loss = (pred - Y_batch) ** 2

            if useMeta:
                loss = loss * weighting

            loss = torch.mean(loss)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #print ('range')
        #print (np.quantile(Y_batch[dataValue > 0].data.numpy(), np.array([0.25, 0.75])))
        #print (np.quantile(Y_batch.data.numpy(), np.array([0.25, 0.75])))


        pred_test, _ = model(X_test)
        pred_test = pred_test[:, 0]

        cor_test = scipy.stats.pearsonr(pred_test.data.numpy(), Y_test.data.numpy())

        if useMeta:
            pred_valid, _ = model(X_valid)
            pred_valid = pred_valid[:, 0]

            #print (pred_valid.shape, Y_valid.shape)
            cor_valid = scipy.stats.pearsonr(pred_valid.data.numpy(), Y_valid.data.numpy())

        pred_train, _ = model(X[:N_test])
        pred_train = pred_train[:, 0]

        cor_train = scipy.stats.pearsonr(pred_train.data.numpy(), Y[:N_test].data.numpy())

        lossInfo = [cor_train[0], cor_test[0]]

        #print ("Scores")
        print ('iter', iter)
        print (cor_train)
        if useMeta:
            print (cor_valid)
            lossInfo = [cor_train[0], cor_valid[0], cor_test[0]]
        print (cor_test)

        trainLoss.append(copy.copy(lossInfo))

        #np.save('./results/trainLoss_someData.npy', np.array(trainLoss))
        #torch.save(model, './models/1_someData.pt')

        torch.save(model, './models/3.pt')


#trainModel()
#quit()



def trainFineTune():

    useMeta = True


    X = loadnpz('./Age/data/allImages.npz')
    Y = loadnpz('./Age/data/allAges.npz')



    Y = Y / 100
    X = np.swapaxes(X, 1, 3)
    X = np.swapaxes(X, 2, 3)

    minAge_test = 0.14
    maxAge_test = 0.21

    #minAge_valid = 0.12
    #maxAge_valid = 0.23
    minAge_valid = 0.14
    maxAge_valid = 0.21



    featureModel = torch.load('./models/2.pt')

    N = Y.shape[0]
    rng = np.random.RandomState(0)
    perm1 = rng.permutation(N)
    X = X[perm1]
    Y = Y[perm1]

    N_test = 1000
    X_test = X[:N_test]
    Y_test = Y[:N_test]
    X = X[N_test:]
    Y = Y[N_test:]
    if useMeta:
        N_valid = 1000
        X_valid = X[:N_valid]
        Y_valid = Y[:N_valid]
        X = X[N_valid:]
        Y = Y[N_valid:]


        validArg = np.argwhere(np.logical_and(Y_valid >= minAge_valid, Y_valid <= maxAge_valid))[:, 0]
        X_valid = X_valid[validArg]
        Y_valid = Y_valid[validArg]

        X_valid = torch.tensor(X_valid).float()
        Y_valid = torch.tensor(Y_valid).float()

        _, X_valid = featureModel(X_valid)


    X = torch.tensor(X).float()
    Y = torch.tensor(Y).float()
    X_test = torch.tensor(X_test).float()
    Y_test = torch.tensor(Y_test).float()




    _, X = featureModel(X)
    X = X.data.numpy()
    X = torch.tensor(X).float()
    _, X_test = featureModel(X_test)





    metaTest = True

    if metaTest:
        testArg = np.argwhere(np.logical_and(Y_test.data.numpy() >= minAge_test, Y_test.data.numpy() <= maxAge_test))[:, 0]
        X_test = X_test[testArg]
        Y_test = Y_test[testArg]


    restrictTrain = False
    if restrictTrain:
        trainArg = np.argwhere(np.logical_and(Y.data.numpy() >= minAge_test, Y.data.numpy() <= maxAge_test))[:, 0]
        X = X[trainArg]
        Y = Y[trainArg]




    N_train = Y.shape[0]



    #model = CNN()

    model = linearOutput()


    sizeTotal = 0
    for param in model.parameters():
        size1 = param.nelement()
        sizeTotal += size1



    #learningRate = 1e-3
    #learningRate = 5e-4
    #learningRate = 3e-4
    #learningRate = 2e-4
    learningRate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)


    iterN = 2000
    batchSize = 100
    Nbatch = N_train // batchSize

    trainLoss = []

    for iter in range(iterN):

        for batchNum in range(Nbatch):

            X_batch = X[batchNum*batchSize:(batchNum+1)*batchSize]
            Y_batch = Y[batchNum*batchSize:(batchNum+1)*batchSize]

            if False:#useMeta:
                dataGrad = findDataGrad(X_batch, Y_batch, model, optimizer, sizeTotal)


                pred_valid, _ = model(X_valid)
                pred_valid = pred_valid[:, 0]
                loss_Valid = torch.mean((pred_valid - Y_valid) ** 2)
                loss_Valid.backward()

                validGrad = calculateGradVector(model, sizeTotal)
                validDir = validGrad
                validDir = validDir.reshape((1, validDir.shape[0]))

                dataValue = np.sum(validDir * dataGrad, axis=1)

                #norm1 = np.sum(validDir**2, axis=1) ** 0.5
                #norm2 = np.sum(dataGrad**2, axis=1) ** 0.5
                #dataValue = dataValue / (norm1 * norm2)

                #plt.scatter(Y_batch.data.numpy(), dataValue)
                #plt.show()
                #quit()

                #weighting = np.zeros(dataValue.shape[0])
                #weighting[dataValue > 0] = 1

                weighting = np.copy(dataValue) / 2
                weighting[weighting<0] = 0
                weighting[weighting>1] = 1

                #print (torch.mean(Y_batch[dataValue > 0]), torch.mean(Y_batch))



                weighting = torch.tensor(weighting).float()




            pred, _ = model(X_batch)
            pred = pred[:, 0]
            loss = (pred - Y_batch) ** 2

            #if useMeta:
            #    loss = loss * weighting

            loss = torch.mean(loss)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #print ('range')
        #print (np.quantile(Y_batch[dataValue > 0].data.numpy(), np.array([0.25, 0.75])))
        #print (np.quantile(Y_batch.data.numpy(), np.array([0.25, 0.75])))


        pred_test, _ = model(X_test)
        pred_test = pred_test[:, 0]

        cor_test = scipy.stats.pearsonr(pred_test.data.numpy(), Y_test.data.numpy())

        if useMeta:
            pred_valid, _ = model(X_valid)
            pred_valid = pred_valid[:, 0]

            #print (pred_valid.shape, Y_valid.shape)
            cor_valid = scipy.stats.pearsonr(pred_valid.data.numpy(), Y_valid.data.numpy())

        pred_train, _ = model(X[:N_test])
        pred_train = pred_train[:, 0]

        cor_train = scipy.stats.pearsonr(pred_train.data.numpy(), Y[:N_test].data.numpy())

        lossInfo = [cor_train[0], cor_test[0]]

        #print ("Scores")
        print ('iter', iter)
        print (cor_train)
        if useMeta:
            print (cor_valid)
            lossInfo = [cor_train[0], cor_valid[0], cor_test[0]]
        print (cor_test)

        trainLoss.append(copy.copy(lossInfo))

        #np.save('./results/trainLoss_someData.npy', np.array(trainLoss))
        #torch.save(model, './models/1_someData.pt')


#trainFineTune()
#quit()


'''
X = torch.zeros((5, 6))
X[:, 5] = 1
X[0, 0] = 1
X[1, 1] = 1
X[2, 2] = 1
X[3, 3] = 1
X[4, 4] = 1

Y = torch.zeros(5)
Y[0] = 1
Y[1] = 1
Y[2] = 1
Y[3] = 1
Y[4] = 1


XX_mat = torch.matmul(X.T, X)

#XX_mat_inv = torch.inverse(XX_mat + (1e-6 * X.shape[0] * torch.eye(XX_mat.shape[0]))  )
XX_mat_inv = torch.inverse(XX_mat + (1e-4 * X.shape[0] * torch.eye(XX_mat.shape[0]))  )
XY_mat = torch.matmul(X.T, Y)

weightBeta = torch.matmul(XX_mat_inv, XY_mat)

weightBeta = weightBeta.reshape((-1, 1))

pred = torch.matmul(X, weightBeta)

print (pred)
print (Y)
quit()
'''




def trainWeightingLinear():

    #useMeta = True

    useMeta = False


    X = loadnpz('./Age/data/allImages.npz')
    Y = loadnpz('./Age/data/allAges.npz')



    Y = Y / 100
    X = np.swapaxes(X, 1, 3)
    X = np.swapaxes(X, 2, 3)

    minAge_test = 0.14
    maxAge_test = 0.21

    #minAge_valid = 0.12
    #maxAge_valid = 0.23
    minAge_valid = 0.14
    maxAge_valid = 0.21

    #minAge_valid = 0.10
    #maxAge_valid = 0.25



    featureModel = torch.load('./models/3.pt')

    N = Y.shape[0]
    rng = np.random.RandomState(0)
    #rng = np.random.RandomState(1)
    #rng = np.random.RandomState(2)
    perm1 = rng.permutation(N)
    X = X[perm1]
    Y = Y[perm1]


    X = torch.tensor(X).float()
    originalPred, X = featureModel(X)
    X = X.data.numpy()
    onesAdd = np.ones((X.shape[0], 1), dtype=float)
    X = np.concatenate((X,onesAdd), axis=1)

    originalPred = originalPred.data.numpy()
    originalPred = torch.tensor(originalPred).float()

    #print (originalPred.shape)
    #quit()


    N_test = 1000
    X_test = X[:N_test]
    Y_test = Y[:N_test]
    originalPred_test = originalPred[:N_test]
    X = X[N_test:]
    Y = Y[N_test:]
    originalPred = originalPred[N_test:]
    if useMeta:
        N_valid = 1000
        X_valid = X[:N_valid]
        Y_valid = Y[:N_valid]
        originalPred_valid = originalPred[:N_valid]
        X = X[N_valid:]
        Y = Y[N_valid:]
        originalPred = originalPred[N_valid:]


        validArg = np.argwhere(np.logical_and(Y_valid >= minAge_valid, Y_valid <= maxAge_valid))[:, 0]
        X_valid = X_valid[validArg]
        Y_valid = Y_valid[validArg]
        originalPred_valid = originalPred_valid[validArg]

        X_valid = torch.tensor(X_valid).float()
        Y_valid = torch.tensor(Y_valid).float()


    X = torch.tensor(X).float()
    Y = torch.tensor(Y).float()
    X_test = torch.tensor(X_test).float()
    Y_test = torch.tensor(Y_test).float()






    metaTest = True

    if metaTest:
        testArg = np.argwhere(np.logical_and(Y_test.data.numpy() >= minAge_test, Y_test.data.numpy() <= maxAge_test))[:, 0]
        X_test = X_test[testArg]
        Y_test = Y_test[testArg]
        originalPred_test = originalPred_test[testArg]


    restrictTrain = True
    if restrictTrain:
        #trainArg = np.argwhere(np.logical_and(Y.data.numpy() >= minAge_valid, Y.data.numpy() <= maxAge_valid))[:, 0]
        trainArg = np.argwhere(np.logical_and(Y.data.numpy() >= minAge_test, Y.data.numpy() <= maxAge_test))[:, 0]
        X = X[trainArg]
        Y = Y[trainArg]
        originalPred = originalPred[trainArg]



    N_train = Y.shape[0]



    #model = CNN()

    #predModel = returnVector(N_train)
    #predModel = linearOutput()
    predModel = weightModel()


    #sizeTotal = 0
    #for param in model.parameters():
    #    size1 = param.nelement()
    #    sizeTotal += size1



    #learningRate = 1e-3
    #learningRate = 5e-4
    #learningRate = 3e-4
    #learningRate = 2e-4
    #learningRate = 1e-2
    learningRate = 1e-3
    optimizer = torch.optim.Adam(predModel.parameters(), lr = learningRate)

    #learningRate = 1e-3
    #learningRate = 1e-4
    #optimizer = torch.optim.RMSprop(predModel.parameters(), lr = learningRate)

    #learningRate = 1e-1
    #optimizer = torch.optim.SGD(predModel.parameters(), lr = learningRate)


    iterN = 100000
    batchSize = 100
    Nbatch = N_train // batchSize

    trainLoss = []

    for iter in range(iterN):


        #print (X[0])
        #print (X[1])
        #quit()

        if useMeta:
            #W = predModel()
            #W = (torch.tanh(W) + 1) / 1
            #W = W.reshape((W.shape[0], 1))

            #W, _ = predModel(X[:, :-1])
            #W = (torch.tanh(W) + 1) / 1

            #X_both = torch.cat((X[:, :-1], Y.reshape((-1, 1)) * 10  ), axis=1)



            #X_both = torch.cat((X[:, :-1] * 0, (Y-0.18).reshape((-1, 1)) , torch.abs(Y-0.18).reshape((-1, 1)) , ((Y-0.18)**2).reshape((-1, 1))  ), axis=1)
            #X_both = torch.cat(( (Y-0.18).reshape((-1, 1)) , torch.abs(Y-0.18).reshape((-1, 1)) , ((Y-0.18)**2).reshape((-1, 1)) , (originalPred-0.18), torch.abs(originalPred-0.18) , (originalPred-0.18) ** 2, torch.abs(Y.reshape((-1, 1)) - originalPred), (Y.reshape((-1, 1)) - originalPred)**2  ), axis=1)
            X_both = torch.cat(( (Y-0.18).reshape((-1, 1)) , (originalPred-0.18) ), axis=1) * 5

            W, _ = predModel(X_both)
            W = (torch.tanh(W) + 1) / 1



        else:
            W = torch.ones((X.shape[0], 1))

        XW = (W * X).T

        XX_mat = torch.matmul(XW, X)

        #print (XX_mat)
        #print ((1e-1 * X.shape[0]))
        #quit()

        ridgeReg = 5e-3
        #ridgeReg = 1e-2
        #ridgeReg = 2e-2
        #ridgeReg = 1e-1

        XX_mat_inv = torch.inverse(XX_mat + (ridgeReg * X.shape[0] * torch.eye(XX_mat.shape[0]))  )
        XY_mat = torch.matmul(XW, Y)

        weightBeta = torch.matmul(XX_mat_inv, XY_mat)

        weightBeta = weightBeta.reshape((-1, 1))


        pred_train = torch.matmul(X, weightBeta)
        pred_train = pred_train[:, 0]

        #print (np.mean(Y.data.numpy()))
        #print (np.mean(pred_train.data.numpy()))
        #print (np.mean(   (pred_train.data.numpy() -  Y.data.numpy())  **2 ))
        #quit()

        cor_train = scipy.stats.pearsonr(pred_train.data.numpy(), Y.data.numpy())


        pred_test = torch.matmul(X_test, weightBeta)
        pred_test = pred_test[:, 0]
        cor_test = scipy.stats.pearsonr(pred_test.data.numpy(), Y_test.data.numpy())

        if useMeta:
            pred_valid = torch.matmul(X_valid, weightBeta)
            pred_valid = pred_valid[:, 0]
            cor_valid = scipy.stats.pearsonr(pred_valid.data.numpy(), Y_valid.data.numpy())




        print (iter)
        print (cor_train)

        if useMeta:
            print (cor_valid)
        print (cor_test)
        #quit()


        if not useMeta:
            quit()

        if useMeta:

            #loss = torch.mean(  (pred_valid - Y_valid) ** 2)
            loss = torch.mean(  torch.abs(pred_valid - Y_valid))

            #pred_valid = pred_valid - torch.mean(pred_valid) + torch.mean(Y_valid)
            #loss = torch.mean(  torch.abs(pred_valid - Y_valid))


            #reg = 0
            #for param in predModel.parameters():
            #    reg += torch.sum(torch.abs(param))

            #loss = loss + (2e-5 * reg)
            #loss = loss + (1e-4 * reg)
            #loss = loss + (1e-3 * reg)
            #loss = loss + (1e-2 * reg)


            optimizer.zero_grad()
            loss.backward()


            torch.nn.utils.clip_grad_value_(predModel.parameters(), 1e-4) #good

            #torch.nn.utils.clip_grad_value_(predModel.parameters(), 2e-5)

            optimizer.step()





trainWeightingLinear()
quit()



def plotTest():

    loss_ours = np.load('./results/trainLoss_ours.npy')
    loss_some = np.load('./results/trainLoss_someData.npy')
    loss_all = np.load('./results/trainLoss_allData.npy')


    plt.plot(loss_ours[:50, -1])
    plt.plot(loss_some[:50, -1])
    plt.plot(loss_all[:50, -1])
    plt.legend(['Ours', 'Restricted', 'All Data'])
    plt.xlabel("Epoch")
    plt.ylabel("Test Correlation")
    plt.show()

#plotTest()
#quit()
#quit()
