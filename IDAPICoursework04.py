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
    
    # print realData.shape
    # print realData
    # print realData.sum(axis=0), theData.shape[1]
    # print realData.shape
    mean = realData.sum(axis=0)/theData.shape[0]


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

    # Coursework 4 task 2 ends here
    return covar
def CreateEigenfaceFiles(theBasis):
    
    # Coursework 4 task 3 begins here

    for idx, basis in enumerate(theBasis):
        SaveEigenface(basis, "PrincipalComponent_" + str(idx) + ".jpg")

    # Coursework 4 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 4 task 4 begins here

    # magnitudes1 = multiply(array(theFaceImage) - array(theMean), theBasis)
    for basis in theBasis:
        magnitudes.append(dot((array(theFaceImage) - array(theMean)), basis))
    # print setdiff1d(magnitudes, magnitudes1)
    # Coursework 4 task 4 ends here
    return array(magnitudes)

def CreatePartialReconstructions(aBasis, aMean, componentMags):
    
    # Coursework 4 task 5 begins here
    # print aMean

    p_x = aMean
    SaveEigenface(array(p_x), "Reconstruction_MeanFace.jpg")

    # p_x = multiply(componentMags[idx], aBasis[0])
    # print p_x

    # SaveEigenface(p_x, "Reconstruction_" + ".jpg")


    for idx, basis in enumerate(aBasis):
        # if idx > 0:
            # recon = ones([len(aBasis[0])])
            
            # for i in range(idx):
            #     recon = multiply(recon, aBasis[i])
                
            # print recon, min(recon), max(recon)
        # p_x = multiply(componentMags[0], recon)
        # print multiply(componentMags[idx], aBasis[idx])
        # p_x = multiply(aBasis[idx], componentMags[idx]) + p_x
        p_x += dot(basis, componentMags[idx].T)

        SaveEigenface(p_x, "Reconstruction_" + str(idx) + ".jpg")


    # Coursework 4 task 5 ends here

def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 4 task 3 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes
    real_data = array(theData)
    # print real_data.shape
    # print real_data
    # print Mean(real_data)
    # print tile(Mean(real_data), (real_data.shape[0], 1)).shape
    
    # print tile(Mean(real_data), (real_data.shape[0], 1))
    # print real_data.shape[0]
    # print tile(Mean(real_data), (real_data.shape[0], 1))
    U = real_data - tile(Mean(real_data), (real_data.shape[0], 1))

    # small_vec = U * U.T
    # print U.T * U
    # print U * U.T
    # print dot(U.T, U).shape
    small_vec = dot(U, U.T)

    # print small_vec
    # small_eig = 1
    w, v = linalg.eig(small_vec)
    # print sum(small_vec, axis=0)

    big_eig = dot(U.T, v)

    # sort_list = 
    print argsort(w)[::-1]
    for item in argsort(w)[::-1]:
        orthoPhi.append(big_eig[:,item]/sqrt(sum(power(big_eig[:,item],2))))

    # for row in big_eig:
    #     pass
        # print sum(power(row, 2))

    print w
    # print 'NORMS'
    # for row in ReadEigenfaceBasis():
    #     print linalg.norm(row)
    # print linalg.norm(ReadEigenfaceBasis())

    for row in orthoPhi:
        print sum(power(row, 2))

    # print U
    # print U.shape
    # print type(U)
    # print max(U), min(U)

    # D = Mean(real_data)
    # print "mean shape", D
    # SaveEigenface(U, 'test.jpg')
    # print setdiff1d(map(int, U), ReadOneImage('MeanImage.jpg'))
    

    
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
# print Covariance(m)
# print cov(m)

# print 

CreateEigenfaceFiles(ReadEigenfaceBasis())

projected_faces = ProjectFace(ReadEigenfaceBasis(), ReadOneImage('MeanImage.jpg'), ReadOneImage('c.pgm'))

CreatePartialReconstructions(ReadEigenfaceBasis(), ReadOneImage('MeanImage.jpg'), projected_faces)

print PrincipalComponents(ReadImages())