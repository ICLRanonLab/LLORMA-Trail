from base.dataset import DatasetManager

#from llorma_p.trainer import main as llorma_parallel_train
from llorma_g.trainer import main as llorma_global_train

import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import metrics
import csv

# 1000x1000 200x200 mean=0 error=10
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--testType", type=str)
parser.add_argument("--size", type=int)
parser.add_argument("--error", type=int)
parser.add_argument("--mean", type=int)
runParams = parser.parse_args()

size = runParams.size
error = runParams.error
mean = runParams.mean
testType = runParams.testType


if __name__ == '__main__':
    # kind = DatasetManager.KIND_MOVIELENS_100K
    # kind = DatasetManager.KIND_MOVIELENS_1M
    # kind = DatasetManager.KIND_MOVIELENS_10M
    # kind = DatasetManager.KIND_MOVIELENS_20M
    #kind = DatasetManager.KIND_PTMATRIX_1000

    #llorma_parallel_train(kind)
    #llorma_global_train(kind)

#MEAN


    i = 1 #for i in range(2):#size
    j = 1 #for j in range(2):#error
    # print(size[1])
    scores = []
    accuracies = []
    k = 1 #for k in range(31):#mean
    # make Answer

    Pat_N = size
    Mat_N = 100
    print("Creating Answers")


    AnswerName = 'Answer'+str(testType)+str(size)+'-'+str(error)+'-'+str(mean)+'.csv'

    print("Answer CREATED!")



    #kind = 'DatasetManager.KIND_PTMATRIX_100_'+str(testType)+str(size)+'_'+str(error)+'_'+str(mean)
    #kind = 'DatasetManager.KIND_PTMATRIX_100' + '_' + str(testType) + str(size) + '_' + str(error) + '_' + str(mean)
    kind = 'ptmatrix-' + str(testType) +'-'+ str(size) +'-'+ str(error) +'-'+ str(mean)
    print(kind)

    #PQ Analyzer
    pNAME = './llorma_g/ptmatrix-basetimes-' + str(size)+'-'+str(error)+'-'+str(mean)+'-p.npy'
    qNAME = './llorma_g/ptmatrix-basetimes-' + str(size) + '-' + str(error) + '-' + str(mean) + '-q.npy'
    p = np.load(pNAME)
    q = np.load(qNAME)

    p = np.concatenate(p)
    q = np.concatenate(q)
    print(p.shape)
    print(p)
    print(q)

    aa = np.zeros((p.shape[0], 1))
    bb = np.zeros((q.shape[0], 1))

    print(type(q))

    sortdP = -np.sort(-p)
    sortdQ = -np.sort(-q)

    print(sortdP)

    critP = sortdP[Pat_N]
    critQ = sortdQ[Pat_N]

    p[p <= critP] = 0
    p[p > critP] = 1

    q[q <= critQ] = 0
    q[q > critQ] = 1

    aa = p.copy()
    bb = q.copy()



    cc = []

    print(pd.value_counts(aa.reshape(Mat_N,)))
    print(pd.value_counts(bb.reshape(Mat_N,)))
    cc = np.matmul(np.reshape(aa, (Mat_N, 1)), np.reshape(bb, (Mat_N, 1)).T)

    AnswerData = pd.read_csv(AnswerName, dtype=int, header=None)
    AD = AnswerData.values
    score = np.zeros(1)
    accuracy = np.zeros(1)
    confuMat = np.zeros(1)
    #print(pd.value_counts(AD.astype(int).reshape(10000, 1)))
    abc = np.array(cc)
    bcd = np.array(AD)
    Mat_S = Mat_N * Mat_N
    print('cc', pd.value_counts(abc.reshape(Mat_S,)))
    print('Answer：', pd.value_counts(bcd.reshape(Mat_S,)))
    # print(cc[0].astype(int))
    # print(AD)
    score = sk.metrics.f1_score(AD.reshape(Mat_S, 1), cc.astype(int).reshape(Mat_S, 1))
    accuracy = sk.metrics.accuracy_score(AD.reshape(Mat_S, 1), cc.astype(int).reshape(Mat_S, 1))
    confuMat = sk.metrics.confusion_matrix(AD.reshape(Mat_S, 1), cc.astype(int).reshape(Mat_S, 1),
                                               labels=[1, 0], normalize='true')

    print(confuMat)
    print("Done Calculation!")

    csv1 = str(testType) + str(size) + '-' + str(error) + 'TPTN' + '.csv'
    res1 = list([testType, size, error, mean, confuMat[0, 0], confuMat[1, 1]])
    res1 = pd.DataFrame(res1)
    res1 = res1.T
    res1.to_csv(csv1, mode='a', header=False, index=False)



