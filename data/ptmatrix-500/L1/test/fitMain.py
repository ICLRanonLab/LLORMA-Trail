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
    """
    # choose cpu or gpu automatically
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #torch.manual_seed(16)
    # partitions
    labels_mateDatas, variance, var_noise = Predicte.myUtils.myData.getL1SpeBaseData(runPams.baseLen, runPams.crType, minusMean, runPams.errorStdBias,
                                                                blockNum, runPams.baseTimes, zn, runPams.xn, yn,
                                                                totalRow, totalCol, overlap,
                                                                replace)
    mateDatas = list(labels_mateDatas[-1][-1])
    mateData = pd.DataFrame(mateDatas[0])#matadata.values
    #mateData_noshuffle = mateData.copy() #1000x1000 w 500x500 unshuffled
    mateData = shuffle(mateData)# hang shuffle
    mateData = shuffle(mateData.T) #lie shuffle
    mateData = mateData.T #ok
    """
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
"""
    # get partitions
    parts = Predicte.myUtils.myData.mateData2Parts(mateData.copy())
    '''
    for part in parts:
        nx = np.sum(part.index<yn)
        ny = np.sum(part.columns<yn)
        print(nx,ny,(nx*ny)/(yn*yn))
    '''
    
    #parts = [ part.abs() for part in parts]
    for _ in range(runPams.n_inner_epochs):
        res = list()
        pool = Pool(os.cpu_count())
        for part in parts:
            res.append(pool.apply_async(Predicte.myUtils.myData.getSamplesRowColStd, args=(part, baseLen)))
        pool.close()
        pool.join()

        # splite samples, samplesArr, rowStdArr, colStdArr from res
        samples = list()
        samplesArr = list()
        # rowStdArr = list()
        # colStdArr = list()
        for r in res:
            tmp = r.get()
            samples.append(tmp[0])
            samplesArr.append(tmp[1])
            # rowStdArr.append(r[2])
            # colStdArr.append(r[3])
        labels = Predicte.myUtils.myData.getLabelFrSamples(samples, runPams.xn, runPams.xn)
        print(pd.value_counts(labels))
        samplesArr = np.stack(samplesArr)
        
        # rowStdArr = np.stack(rowStdArr)
        # colStdArr = np.stack(colStdArr)

        # get bases matrix
        #basesMtrx, baseTypeNumAfterKmean, baseIdAfterKMeans = Predicte.myUtils.myData.getBasesMtrxAfterKmean(n_clusters)
        basesMtrx, baseTypeNumAfterKmean, baseIdAfterKMeans = Predicte.myUtils.myData.getBasesMtrxAfterKmean_mul(baseLen) 
        
        # get row and col feature map:
        rowFeatureMap = np.matmul(samplesArr, (basesMtrx.iloc[:, 0:baseLen].values.T))
        # colFeatureMap = np.matmul(samplesArr.transpose((0, 1, 3, 2)), (basesMtrx.iloc[:, 0:7].values.T))

        # normalize row and col by std from original 50*50's row and col std
        # rowFeatureMap = np.true_divide(rowFeatureMap, rowStdArr)

        #sort the col small -> big
        rowFeatureMap = -(np.sort(-(rowFeatureMap), axis=2))

        # normalize col by std from original 50*50' col
        # colFeatureMap = np.true_divide(colFeatureMap, colStdArr)
        # colFeatureMap = -np.sort(-colFeatureMap, axis=2)

        # resort them by their mean
        #delete the first 2 rows
        rowFeatureMap = Predicte.myUtils.myData.getResortMeanFeatureMap(rowFeatureMap[:, :, 2:baseLen, :])
        # colFeatureMap = Predicte.myUtils.myData.getResortMeanFeatureMap(colFeatureMap)

        # row and col max pooling 5*1652 -> 5*16
        rowFeatureMap = Predicte.myUtils.myData.myMaxPooling(rowFeatureMap, baseTypeNumAfterKmean)
        # colFeatureMap = Predicte.myUtils.myData.myMaxPooling(colFeatureMap, baseTypeNumAfterKmean)
        # featureMap = np.stack((rowFeatureMap, colFeatureMap), axis=2)

        # sort the rows small -> big
        rowFeatureMap = -(np.sort(-(rowFeatureMap), axis=3))[:, :, :, :]
        
        # first 5 cols - last 5 cols 
        rowFeatureMap = rowFeatureMap[:, :, :, 0:baseLen] - rowFeatureMap[:, :, :, (16-baseLen):16]
        print(rowFeatureMap.shape) 
        '''
        featureMean = np.mean(rowFeatureMap, axis=3)
        zn, xn, yn = featureMean.shape
        featureMean = np.reshape(featureMean, (zn, xn, yn, 1))
        featureStd = np.std(rowFeatureMap, axis=3)
        zn, xn, yn = featureStd.shape
        featureStd = np.reshape(featureStd, (zn, xn, yn, 1))
        rowFeatureMap = np.true_divide(rowFeatureMap, featureMean)
        '''

        #rowFeatureMap_np = rowFeatureMap.copy()
        labels = torch.tensor(labels).long()
        rowFeatureMap = torch.tensor(rowFeatureMap).float()
        rowFeatureMap = rowFeatureMap.view(rowFeatureMap.size()[0] * rowFeatureMap.size()[1], rowFeatureMap.size()[2],
                                           rowFeatureMap.size()[3])

        # rowFeatureMap = torch.rand(100, 7, 16)
        torch.manual_seed(baseLen)
        # optFeatureMap data
        currentDate = str(time.strftime("%Y%m%d", time.localtime()))
        pathName = "/N/project/zhangclab/pengtao/myProjectsDataRes/20200113Predicte/data/l1/preTrain/base"+str(baseLen)+"/addAll/"
        #fileName = pathName + "FCNModel_preTrain_16_5_5_1000_0.0001_20200917.pkl"
        fileName = pathName + "FCNModel_preTrain_16_300_0.001_20200926.pkl"
        net = torch.load(fileName, map_location=device)
        net = net.to(device)
        
        '''
        z, predLabels = Predicte.myUtils.myTrainTest.train_test_FCN(
            rowFeatureMap, labels, net, device, optimizer, lossFunc, runPams)
        '''
        
        acc, SE, predLabelsByFCN = Predicte.myUtils.myTrainTest.fit_FCN(rowFeatureMap, labels, net, device, runPams)
        #print(predLabelsByFCN)
        print(acc)
        print(SE)
        #sys.exit()
        '''
        kmeans_estimator = KMeans(n_clusters=2, random_state=0).fit(z)
        kmeans_label_pred = kmeans_estimator.labels_
        predLabels = kmeans_label_pred
        print(predLabels)
        '''
        
        predLabels = [pl[1] for pl in predLabelsByFCN]
        predLabels = np.concatenate(predLabels)
        
        counts = pd.value_counts(predLabels)
        print(counts)

        if (len(predLabels) != (len(samples)*len(samples[0]))):
            print("size match error!")
            return()
        
        labelType =np.sort(np.unique(predLabels))
        print(labelType)
        classNum = len(labelType)
        predLabels = np.resize(predLabels, (len(samples), len(samples[0]), 1))

        # get update row and col indices
        # initial the new empty samples list
        allNewSamples = list()
        for _ in range(classNum):
            allNewSamples.append([])

        # re generate the samples by their generated label
        sampleSetNum = len(samples)
        samplesNum = len(samples[0])
        for i in range(sampleSetNum):
            for j in range(samplesNum):
                label = predLabels[i][j]
                idx = np.where(labelType == label.item())[0][0]
                allNewSamples[idx].append(samples[i][j])
        
        tmpSamples = list()
        for i in range(len(counts)):
            maxLabel = counts.idxmax()
            maxIdx = np.where(labelType==maxLabel)[0][0]
            tmpSamples.append([maxLabel, allNewSamples[maxIdx]])
            counts[maxLabel] = -1
        print(counts)
        #maxLabel = counts.idxmax()
        #maxIdx = np.where(labelType==maxLabel)[0][0]
        #tmpSamples.append([maxLabel, allNewSamples[maxIdx]])
        
    
        if tmpSamples[0][0] > tmpSamples[1][0]:
            tmpSamples[0], tmpSamples[1] = tmpSamples[1], tmpSamples[0]
        

        #idxMax = 1
        # get new expand samples from mateData
        # test = Predicte.myUtils.myData.getNewPart(allNewSamples[0], mateData)
        allNewSamples = tmpSamples
        pool = Pool(os.cpu_count())
        tmpResults = list()
        for samples in allNewSamples:
            tmpResults.append(pool.apply_async(Predicte.myUtils.myData.getNewPart, args=(samples, mateData, runPams.xn)))
        pool.close()
        pool.join()

        # get new partitions
        newParts = list()
        for res in tmpResults:
            newParts.append(res.get())
        parts = newParts
    # caculate the match degree
    matchLabel = list()
    for label, newPart in newParts:
        print(label)
        if len(newPart) < yn:
            continue
        if len(newPart) == 0:
            matchLabel.append("Nan")
            continue
        #print(np.sort(newPart.index))
        #print(np.sort(newPart.columns))
        matchRowLen = np.sum(list(map(lambda x: x < yn, newPart.index)))
        matchColLen = np.sum(list(map(lambda x: x < yn, newPart.columns)))
        accuracy = ((matchRowLen * matchColLen) / (yn * yn)) * 100
        accuracy = np.around(accuracy, decimals=2)
        matchLabel.append(accuracy)
    print(matchLabel)
    pattern_acc = np.max(matchLabel)
    bgNoise_acc = 100 - np.min(matchLabel)
    #matchLabel = np.sort(matchLabel)
    # matchLabel = ','.join(str(l) for l in matchLabel)

    # output the results
    res = list()
    res.append(str(runPams.xn))
    res.append(str(np.around(runPams.baseTimes, decimals=2)))
    res.append(str(np.around(runPams.errorStdBias, decimals=2)))
    res.append(str(np.around(variance, decimals=2)[0]))
    res.append(str(np.around(var_noise, 2)))
    res.append(acc)
    res.append(SE)
    res.append(str(np.around(pattern_acc, 2)))
    res.append(str(np.around(bgNoise_acc, 2)))
    '''
    for label in matchLabel:
        label = str(np.around(label, 2))
        res.append(label)
    '''
    res = pd.DataFrame(res)
    res = res.T
    print(res)
    #sys.exit()
    pathName = "/N/project/zhangclab/pengtao/myProjectsDataRes/20200113Predicte/results/l1UniformWholeTest_small/block1/csvRes/base"+str(baseLen)+"/addNoRes/"
    if not os.path.exists(pathName):
        os.makedirs(pathName)
    fileName = pathName + "finalRes_U_fit_base"+str(baseLen)+"_"+currentDate+"_"+ runPams.testType+".csv"
    res.to_csv(fileName, mode="a", index=False, header=False)
    print("end")
    return ()

"""
# run main
if __name__ == "__main__":
    main()
