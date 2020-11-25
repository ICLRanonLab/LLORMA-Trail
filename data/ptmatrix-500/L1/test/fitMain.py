# -*- coding: utf8 -*

# system lib
import argparse
import sys
import time
import os
from multiprocessing import Pool

# third part lib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# my libs
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath("../../"))

#import Predicte.myModules
#import Predicte.myUtils.myTrainTest
import Predicte.myUtils.myData
import torch

# parameters
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--n_inner_epochs", type=int, default=1)
if torch.cuda.is_available():
    parser.add_argument("--batch_size", type=int, default=512)
else:
    parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--n_cpu", type=int, default=os.cpu_count())
parser.add_argument("--minusMean", type=int, default=1)
parser.add_argument("--xn", type=int, default=20) #size
parser.add_argument("--crType", type=str, default="uniform")
parser.add_argument("--baseTimes", type=int, default=0)  #mean x/10(calculated later)  0,1,2,3,...
parser.add_argument("--errorStdBias", type=int, default=0) #error x/10 use
parser.add_argument("--testType", type=str, default="testTime")
parser.add_argument("--baseLen", type=int, default=7)
runPams = parser.parse_args()

def getFCNPams(rowNum, colNum, device, lr):
    fcn = Predicte.myModules.FCN(rowNum=rowNum, colNum=colNum)
    fcn = fcn.to(device)
    optimizer = torch.optim.Adam(fcn.parameters(), lr=lr)
    lossFunc = nn.CrossEntropyLoss()
    return (fcn, optimizer, lossFunc)


def getCNNPams(zn, xn, yn, device, lr):
    cnnXout = Predicte.myUtils.myData.getCNNOutSize(xn, 3, 2)
    cnnYout = Predicte.myUtils.myData.getCNNOutSize(yn, 3, 2)
    cnn = Predicte.myModules.CNN(inChannels=zn,
                                 kernels=6,
                                 kernelSize=2,
                                 outSize=16 * cnnXout * cnnYout)
    cnn = cnn.to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
    lossFunc = nn.CrossEntropyLoss()
    return (cnn, optimizer, lossFunc)


# end

def getAEPams(xn, yn, device, lr):
    AE = Predicte.myModules.AutoEncoder(xn, yn)
    AE = AE.to(device)
    optimizer = torch.optim.Adam(AE.parameters(), lr=lr)
    lossFunc = nn.MSELoss()
    return (AE, optimizer, lossFunc)


# end

def getVAEPams(xn, yn, device, lr):
    VAE = Predicte.myModules.VAE(xn, yn)
    VAE = VAE.to(device)
    optimizer = torch.optim.Adam(VAE.parameters(), lr=lr)
    lossFunc = nn.MSELoss()
    return (VAE, optimizer, lossFunc)


# end

def getGANPams(xn, yn, device, lr):
    # G parts
    G = Predicte.myModules.Generator(xn, yn)
    G = G.to(device)
    G_optimizer = torch.optim.Adam(G.parameters(), lr=lr)

    # D parts
    D = Predicte.myModules.Discriminator(xn, yn)
    D = D.to(device)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=lr)
    lossFunc = nn.BCELoss()
    return (G, G_optimizer, D, D_optimizer, lossFunc)


# end


def main():
    # parameters
    mean = 0
    minusMean = 1
    blockNum = 1
    replace = 1
    zn = 1
    yn = runPams.xn
    totalRow = 10000 #100 500 1000 2000 5000 10000
    totalCol = totalRow #100 500 1000 2000 5000 10000
    overlap = 0
    probType = "l1"
    LK=1
    runPams.xn = int(totalCol/2) # Nov.19
    # size = runPams.xn
    size = runPams.xn
    errorN = runPams.errorStdBias
    mean = runPams.baseTimes

    yn = runPams.xn  # to make it a square mat
    #get the std 
    c = np.random.uniform(low=0.0, high=1.0, size=(runPams.xn,LK))
    r = np.random.uniform(low=0.0, high=1.0, size=(LK, yn)) # xn was yn
    part = np.matmul(c,r)
    part_std = np.std(part)
    
    #the number of base 
    baseLen = runPams.baseLen
    runPams.errorStdBias = runPams.errorStdBias / 10
    runPams.errorStdBias = runPams.errorStdBias * (part_std)
    runPams.baseTimes = runPams.baseTimes / 10

    runPams.baseTimes = runPams.baseTimes * (part_std)
    #NovUpdate
    bgNoise = np.random.normal(loc=0.0, scale=part_std, size=(totalRow, totalCol))

    bgNoise = bgNoise - (np.mean(bgNoise, axis=1).reshape(bgNoise.shape[0], 1))

    bgNoise_std = np.std(bgNoise, axis=1).reshape(bgNoise.shape[0], 1)

    bgNoise_std[bgNoise_std == 0] = 1

    bgNoise = np.true_divide(bgNoise, bgNoise_std)

    print(bgNoise[0:5, 0:5])

    print(bgNoise.shape)

    part = part - (np.mean(part, axis=1).reshape(part.shape[0], 1))

    part_std = np.std(part, axis=1).reshape(part.shape[0], 1)

    part_std[part_std == 0] = 1

    part = np.true_divide(part, part_std)


    error = np.random.normal(loc=0, scale=runPams.errorStdBias, size=(runPams.xn, runPams.xn))

    part = part + error + runPams.baseTimes

    bgNoise[0:runPams.xn, 0:runPams.xn] = part

    mateData = bgNoise
    mateData = pd.DataFrame(mateData)

    mateData = shuffle(mateData)

    mateData = shuffle(mateData.T)

    mateData = mateData.T

    print(runPams.baseTimes)
    print(runPams.errorStdBias)

    print(mateData.iloc[0:5, 0:5])
    #mateData = shuffle(mateData.T)
    # Data = mateData.values #OOKK
    #import os
    #pathName = './'+ 'ptmatrix-1000'+'-'+str(size)+'-'+str(error)+'-'+str(mean)
    #os.mkdir(pathName)
    name = str(runPams.testType)+str(totalCol)+'-'+str(errorN)+'-'+str(mean)+'.csv'
    # name = str(runPams.testType)+str(mean)+'-'+str(errorN) + '-' + str(size)+'.csv'
    # name = str(runPams.testType) + str(errorN) + '-' + str(mean) + '-' + str(size) + '.csv'
    #finalPath = pathName + '/' + name
    #mateData.to_csv(path_or_buf=name)
    print(name)
    mateData.to_csv(path_or_buf=name)
    sys.exit()


    # mateData = pd.DataFrame(np.random.rand(1000, 1000))

# run main
if __name__ == "__main__":
    main()
