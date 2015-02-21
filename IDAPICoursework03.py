#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import *
from pprint import pprint

#
# Coursework 1 begins here
#

# Function to compute the prior distribution of the variable root from the data set
def Prior(theData, root, noStates):
    prior = zeros((noStates[root]), float )
# Coursework 1 task 1 should be inserted here
    
    # iterate through data
    for entry in theData:

        # add 1 to the indexed variable
        prior[entry[root]] += 1

    # divide by the number of samples for the prior distribution - ie normalise to sum 1
    prior = prior/len(theData)
# end of Coursework 1 task 1
    return prior


# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = zeros((noStates[varC], noStates[varP]), float )
# Coursework 1 task 2 should be inserted here

    # iterate through data
    for entry in theData:
        
        # idexing the child state first, then the parent, following the structure of the initialised variable cPT
        cPT[entry[varC], entry[varP]] += 1

    cPT = cPT/cPT.sum(axis=0)
    
# end of coursework 1 task 2
    return cPT


# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float )
#Coursework 1 task 3 should be inserted here 

    for entry in theData:
        jPT[entry[varRow], entry[varCol]] += 1

    # Since it's a JPT, we divide by the total number of samples taken
    jPT = jPT/len(theData)

# end of coursework 1 task 3
    return jPT
#
# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
#Coursework 1 task 4 should be inserted here 
    
    # Normalise each column 
    aJPT = aJPT/aJPT.sum(axis=0)

# coursework 1 taks 4 ends here
    return aJPT

#
# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes): 
    rootPdf = zeros((naiveBayes[0].shape[0]), float)
# Coursework 1 task 5 should be inserted here
    prior = naiveBayes[0]

    for root_state_no, p in enumerate(prior):

        # first instantiate with the prior
        qs = [p]

        # iterate through each child, getting data about each one and append to list
        for child_idx, instan in enumerate(theQuery):
            qs.append(naiveBayes[child_idx+1][instan][root_state_no])
        
        # compute the posterior by multiplying all the children and prior together
        result = 1
        for i in qs:
            result = result*i
        
        # set the result back to the rootPdf
        rootPdf[root_state_no] = result

    # Normalise rootPdf, but removing Nans if they come up (happens if the sum is 0)
    rootPdf = nan_to_num(rootPdf/sum(rootPdf))

# end of coursework 1 task 5
    return rootPdf
#
# End of Coursework 1
#
# Coursework 2 begins here
#
# Calculate the mutual information from the joint probability table of two variables
def MutualInformation(jP):
    mi=0.0
# Coursework 2 task 1 should be inserted here
    
    # marinalise the matrix
    row_sum = []
    col_sum = []

    for row in jP:
        row_sum.append(sum(row))

    for col in jP.T:
        col_sum.append(sum(col))

    # Create the matrix for each row and column
    for row_idx, row in enumerate(jP):
        for col_idx, col in enumerate(row):
            if col*row_sum[row_idx]*col_sum[col_idx] != 0 :
                logval = log2(col/(row_sum[row_idx]*col_sum[col_idx]))

                mi += col*logval

# end of coursework 2 task 1
    return mi
#
# construct a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables,noVariables))
# Coursework 2 task 2 should be inserted here

    for i in range(0, noVariables):
        for j in range(0, noVariables):
            MIMatrix[i][j] = MutualInformation(JPT(theData, i, j, noStates))

# end of coursework 2 task 2
    return MIMatrix
# Function to compute an ordered list of dependencies 
def DependencyList(depMatrix):
    depList=[]
# Coursework 2 task 3 should be inserted here

    for row_idx, row in enumerate(depMatrix):
        for col_idx, col in enumerate(row):

            # we don't include rows which are 'self' mutual information
            if col_idx != row_idx:

                flag = True
                for dep_list_row in depList:

                    # checking to ensure we don't print the same row twice
                    if [col_idx, row_idx] == dep_list_row[1:]:
                        flag = False

                if flag:
                    depList.append([col, row_idx, col_idx])

    # Sort the list so they are in dependence order (ie most dependent at the top)
    depList = sorted(depList, key=lambda depList: depList[0], reverse=True)

# end of coursework 2 task 3
    return array(depList)
#
# Functions implementing the spanning tree algorithm
# Coursework 2 task 4

def SpanningTreeAlgorithm(depList, noVariables):

    spanningTree = []

    # Recursive function which tries to find path from one node to another
    # Inspired from https://www.python.org/doc/essays/graphs/
    def find_path(connections, start, end, path=[]):
        path = path + [start]
        
        if start == end:
            return path
        
        for node in connections[start]:
            if node not in path:
                newpath = find_path(connections, node, end, path)
                if newpath:
                    return newpath

        return None


    # Just in case it isn't already sorted
    depList = sorted(depList, key=lambda depList: depList[0], reverse=True)

    # Dictionary of the direction connections from a node
    connections = {}

    for i in range(0, noVariables):
        connections[i] = []

    # We iterate through each row of the dependency list. If the two nodes are not directly or 
    # indirectly connected, we save them to the spanningTree. This works because the list is 
    # in order, so we know that it is maximally weighted.
    for row in depList:

        # ie if they aren't directly or indirectly connected - if they are, this will cause a loop
        if not find_path(connections, row[1], row[2]):

            # Append to our list of edges
            spanningTree.append(row)

            # Append to our dict of directly connected nodes
            connections[row[1]].append(row[2])
            connections[row[2]].append(row[1])

    return array(spanningTree)
#
# End of coursework 2
#
# Coursework 3 begins here
#
# Function to compute a CPT with multiple parents from he data set
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT_2(theData, child, parent1, parent2, noStates):
    cPT = zeros([noStates[child],noStates[parent1],noStates[parent2]], float )
# Coursework 3 task 1 should be inserted here

    for entry in theData:
        cPT[entry[child], entry[parent1], entry[parent2]] += 1

    cPT = numpy.divide(cPT, cPT.sum(axis=0))

    # correction to remove NaNs
    cPT[numpy.isnan(cPT)] = 0

# End of Coursework 3 task 1           
    return cPT
#
# Definition of a Bayesian Network
def ExampleBayesianNetwork(theData, noStates):
    arcList = [[0],[1],[2,0],[3,2,1],[4,3],[5,3]]
    cpt0 = Prior(theData, 0, noStates)
    cpt1 = Prior(theData, 1, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT_2(theData, 3, 2, 1, noStates)
    cpt4 = CPT(theData, 4, 3, noStates)
    cpt5 = CPT(theData, 5, 3, noStates)
    cptList = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5]
    return arcList, cptList
# Coursework 3 task 2 begins here

def HepitisCBayesianNetwork(theData, noStates):
    arcList = [[0], [1], [2, 0], [3, 4], [4, 1], [5, 4], [6, 1], [7, 0, 1], [8, 7]]
    
    cpt0 = Prior(theData, 0, noStates)
    cpt1 = Prior(theData, 1, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT(theData, 3, 4, noStates)
    cpt4 = CPT(theData, 4, 1, noStates)
    cpt5 = CPT(theData, 5, 4, noStates)
    cpt6 = CPT(theData, 6, 1, noStates)
    cpt7 = CPT_2(theData, 7, 0, 1, noStates)
    cpt8 = CPT(theData, 8, 7, noStates)

    cptList = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5, cpt6, cpt7, cpt8]

    return arcList, cptList
# end of coursework 3 task 2
#
# Function to calculate the MDL size of a Bayesian Network
def MDLSize(arcList, cptList, noDataPoints, noStates):
    mdlSize = 0.0
# Coursework 3 task 3 begins here
    logN = log2(noDataPoints)/2
    
    B = 0

    for entry in arcList:

        # shorthand variable to avoid repetition and potential typos
        the_cpt_list = cptList[entry[0]]

        if len(entry) == 1:
            B += len(the_cpt_list) - 1
    
        if len(entry) == 2:
            B += (len(the_cpt_list) - 1) * the_cpt_list.shape[1]
        
        if len(entry) == 3:
            B += (len(the_cpt_list) - 1) * the_cpt_list.shape[1] * the_cpt_list.shape[2]
    
    mdlSize = B * logN

# Coursework 3 task 3 ends here 
    return mdlSize 
#
# Function to calculate the joint probability of a single data point in a Network
def JointProbability(dataPoint, arcList, cptList):
    jP = 1.0
# Coursework 3 task 4 begins here

    for entry in arcList:
        if len(entry) == 1:
            jP = jP * cptList[entry[0]][dataPoint[entry[0]]]
        
        elif len(entry) == 2:
            jP = jP * cptList[entry[0]][dataPoint[entry[0]]][dataPoint[entry[1]]]
            
        elif len(entry) == 3:
            jP = jP * cptList[entry[0]][dataPoint[entry[0]]][dataPoint[entry[1]]][dataPoint[entry[2]]]


# Coursework 3 task 4 ends here 
    return jP
#
# Function to calculate the MDLAccuracy from a data set
def MDLAccuracy(theData, arcList, cptList):
    mdlAccuracy=0
# Coursework 3 task 5 begins here
    
    # sum the log2 of each probability
    for entry in theData:
        mdlAccuracy += log2(JointProbability(entry, arcList, cptList)) 

# Coursework 3 task 5 ends here 
    return mdlAccuracy

def MaxScoreNetwork(theData, noDataPoints, noStates, noRoots, arcList, cptList):

    # Used later on to save the best lists and ensure they are not modified later
    from copy import deepcopy

    # Function to recalculate the cpt depending on the number of parents
    def recalculate_cpt(modified_arc):
        if len(modified_arc) == 1:
            return Prior(theData, modified_arc[0], noStates)
        
        elif len(modified_arc) == 2:
            return CPT(theData, modified_arc[0], modified_arc[1], noStates)

    # Pick high number - hopefully we will never go this high with our calculations!
    best_score = 1e10000

    # iterate through all of the arcs in the list
    for i in range(len(arcList)):

        # save the arc and cpt in its original state, so they can be restored after testing
        arc_piece = arcList[i]
        cpt_piece = cptList[i]

        # in the case of a singly connected node, we remove its parent
        if len(arcList[i]) == 2:

            # replace arclist and cptlist with new values for testing
            arcList[i] = [arcList[i][0]]
            cptList[i] = recalculate_cpt(arcList[i])
            
            # print arcList
            score = MDLSize(arcList, cptList, noDataPoints, noStates) - MDLAccuracy(theData, arcList, cptList)
            # print score

            if score < best_score:
                best_score = score
                best_arclist = deepcopy(arcList)
                best_cptlist = deepcopy(cptList)

            arcList[i] = arc_piece
            cptList[i] = cpt_piece

        if len(arcList[i]) == 3:

            for j in range(1,3):
            
                arcList[i] = [arcList[i][0], arcList[i][j]]
                cptList[i] = recalculate_cpt(arcList[i])

                # print arcList
                score = MDLSize(arcList, cptList, noDataPoints, noStates) - MDLAccuracy(theData, arcList, cptList)
                # print score
                
                if score < best_score:
                    best_score = score
                    best_arclist = deepcopy(arcList)
                    best_cptlist = deepcopy(cptList)

                arcList[i] = arc_piece
                cptList[i] = cpt_piece

    return best_score, best_arclist, best_cptlist
#
# End of coursework 2
#
# Coursework 3 begins here
#
def Mean(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    mean = []
    # Coursework 4 task 1 begins here



    # Coursework 4 task 1 ends here
    return array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    covar = zeros((noVariables, noVariables), float)
    # Coursework 4 task 2 begins here


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


arcList, cptList = HepitisCBayesianNetwork(theData, noStates)
hepc_mdlsize = MDLSize(arcList, cptList, noDataPoints, noStates)

AppendString("results.txt", "The MDLSize of the network of the Hepatitis C data set")
AppendString("results.txt", hepc_mdlsize)
AppendString("results.txt","") #blank line

hepc_mdlacc = MDLAccuracy(theData, arcList, cptList)

AppendString("results.txt", "The MDLAccuracy of the network of the Hepatitis C data set")
AppendString("results.txt", hepc_mdlacc)
AppendString("results.txt","") #blank line

AppendString("results.txt", "The MDLScore of the network of the Hepatitis C data set")
AppendString("results.txt", hepc_mdlsize - hepc_mdlacc)
AppendString("results.txt","") #blank line

best_score, best_arclist, best_cptlist = MaxScoreNetwork(theData, noDataPoints, noStates, noRoots, arcList, cptList)
AppendString("results.txt", "The MDLScore of the best network (of all the networks with one arc removed) of the Hepatitis C data set")
AppendString("results.txt", best_score)

# print JointProbability([0, 2, 0, 9, 8, 6, 6, 4, 1], arcList, cptList)



# numpy.savetxt("foo.csv", DependencyMatrix(datain, noVariables, noStates), delimiter=",")


