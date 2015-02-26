#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import *
from pprint import pprint

#
# Coursework 4 begins here
#
def Mean(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    mean = []
    # Coursework 4 task 1 begins here
    print realData.shape
    print realData
    # print realData.sum(axis=0), theData.shape[1]
    mean = realData.sum(axis=0)/theData.shape[1]


    # Coursework 4 task 1 ends here
    return array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    covar = zeros((noVariables, noVariables), float)
    # Coursework 4 task 2 begins here
    noSamples = theData.shape[0]

    m = Mean(theData)
    print m
    
    def get_var(idx):
        tot = 0.0
        # exit()
        for i in realData[:,idx]:
            # print i, m[idx]
            tot += i - m[idx]
            # print tot

        return tot

    for row_idx, row in enumerate(covar):
        # this_row = theData[row_idx]
        for idx, item in enumerate(row):
            print item, idx, row_idx
            # print get_var(row_idx), get_var(idx)
            s = 0.0
            print range(noSamples)
            for i in range(noSamples):
                print realData[row_idx, i], realData[idx, i]
                s += (realData[i, row_idx] - m[row_idx])*(realData[i, idx] - m[idx])

            # item = (get_var(row_idx) * get_var(idx)) #/(noSamples - 1)
            covar[row_idx, idx] = s/(noSamples - 1)
            # item = row_idx + idx
            # print item, 'row', row_idx, 'column', idx
        print row

    # Coursework 4 task 2 ends here
    return covar
def CreateEigenfaceFiles(theBasis):
    adummystatement = 0 #delete this when you do the coursework
    # Coursework 4 task 3 begins here

    # Coursework 4 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 4 task 4 begins here

    # Coursework 4 task 4 ends here
    return array(magnitudes)

def CreatePartialReconstructions(aBasis, aMean, componentMags):
    adummystatement = 0  #delete this when you do the coursework
    # Coursework 4 task 5 begins here

    # Coursework 4 task 5 ends here

def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 4 task 3 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes

    
    # Coursework 4 task 6 ends here
    return array(orthoPhi)

#
# main program part for Coursework 1
#
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)
AppendString("results.txt","Coursework Three Results by txl11")
AppendString("results.txt","") #blank line

# test_data = [ array([[1, 2, 5], [3, 4, 5]]), random.rand(3, 5), random.rand(5,3), random.rand(8, 8) ]

# for t in test_data:
#     print Mean(t)
#     # print average(t, axis=1), "\n"x
m = array([[-1, 1, 2], [ -2, 3, 1], [4, 0,3]])
print Covariance(m)
print cov(m)

# print JointProbability([0, 2, 0, 9, 8, 6, 6, 4, 1], arcList, cptList)



# numpy.savetxt("foo.csv", DependencyMatrix(datain, noVariables, noStates), delimiter=",")


