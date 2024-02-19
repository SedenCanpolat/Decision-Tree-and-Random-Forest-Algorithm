
import pandas as pd
import numpy as np
import math

from warnings import filterwarnings
filterwarnings("ignore", category=FutureWarning)

TrainSet = pd.read_csv("trainSet.csv")
TestSet = pd.read_csv("testSet.csv")

columnHeaders = TrainSet.columns
TrainSetValues = TrainSet.values


# checks the purity of the data
def isItPure(data):
    classes = data[:, -1]
    if len(np.unique(classes)) == 1: #pure
        return True
    else:
        return False


# classifies the data
def classify(data):
    classes = data[:, -1]
    uniqueNames, uniqueCounts = np.unique(classes, return_counts = True)
    return uniqueNames[np.argmax(uniqueCounts)]
    

# splits the data as below and above
def splitData(data, splitValue, splitedColumn):
    columnValues = data[:, splitedColumn]
    if(not isinstance(columnValues[0], str)): # for numerical values
        aboveData = data[columnValues > splitValue]
        belowData = data[columnValues <= splitValue]
        
    else:
        aboveData = data[(columnValues == splitValue)] # for nominal values
        belowData = data[(columnValues != splitValue)]
    
    return belowData, aboveData
#splitData(TrainSetValues, "a", 0)


# finds possible splits
def findPossibleSpilts(data):
    possibleSpilts = []
    _, columnCount = TrainSet.shape

    columnRange = range(columnCount-1)    
    for columnIndex in columnRange:
        columnValue = data[:,columnIndex]
        uniqueColumnValues = np.unique(columnValue)

        if(not isinstance(uniqueColumnValues[0], str)): # for numerical values
            columnsplits = []
            for i in range(len(uniqueColumnValues)-1):
                possibleSpilt = (uniqueColumnValues[i] + uniqueColumnValues[i+1])/2
                columnsplits.append(possibleSpilt)
        
            possibleSpilts.append(columnsplits)
        else: # for nominal values
            possibleSpilts.append(uniqueColumnValues)
    return possibleSpilts
#print(findPossibleSpilts(TrainSetValues))

# calculates entropy
def calculateEntropy(data):
    entropy = 0
    classes = data[:, -1]
    _,uniqueCounts = np.unique(classes,return_counts = True)
    for uniqueCount in uniqueCounts:
        entropy += ((uniqueCount/len(classes)) * (- math.log2(uniqueCount/len(classes))))
    return entropy
#print(calculateEntropy(TrainSetValues))


# calculates overall entropy
def calculateOverallEntropy(data, aboveData, belowData):
    classesCount = len(data[:,-1])
    overallEntropy = (len(aboveData)/classesCount * calculateEntropy(aboveData)) + (len(belowData)/classesCount * calculateEntropy(belowData))
    return overallEntropy
#calculateOverallEntropy(TrainSetValues, aboveData, belowData)



#compares entropy then finds value for the best split and column
def findBestSplit(data):
    currentEntropy = 999
    possibleSplits = findPossibleSpilts(data)
  
    for columnIndex in range(len(possibleSplits)):
        columnValues = possibleSplits[columnIndex]
        for value in columnValues:
                aboveData, belowData = splitData(data, splitValue=value, splitedColumn=columnIndex)
                overallEntropy = calculateOverallEntropy(data, aboveData=aboveData, belowData=belowData)
                if overallEntropy < currentEntropy:
                        currentEntropy = overallEntropy
                        bestSplitValue = value
                        bestSplitColumn = columnIndex
            
    return bestSplitValue, bestSplitColumn
#print(findBestSplit(TrainSetValues))


# creates decision tree
def createDecisionTree(data, depthLimit, depth=0):
                
        if depth == 0:  
            data = data.values
    
        if isItPure(data) or depth >= depthLimit: #base
            classification = classify(data)
            return classification
        else: #recursion
            depth += 1

            bestSplitValue, bestSplitColumn = findBestSplit(data)
            belowData, aboveData = splitData(data, splitValue=bestSplitValue, splitedColumn=bestSplitColumn)
            HeaderName = columnHeaders[bestSplitColumn]

            if len(aboveData) == 0 or len(belowData) == 0:
                classification = classify(data)
                return classification

            if(not isinstance(bestSplitValue, str)): # for numerical values
                question = "{} <= {}".format(HeaderName, bestSplitValue)
            else: # for nominal values
                question = "{} != {}".format(HeaderName, bestSplitValue)   
            subTree = {question: []}

            yesAnswer = createDecisionTree(belowData, depthLimit, depth)
            noAnswer = createDecisionTree(aboveData, depthLimit, depth)

            if yesAnswer == noAnswer:
                noAnswer = yesAnswer
            else:
                subTree[question].append(yesAnswer)
                subTree[question].append(noAnswer) 
                
        return subTree


# asks questions about the data for analyzing the results
def findAnswer(subtree, data):
    question = list(subtree.keys())[0]
   
    header, operator, value = question.split()

    if operator == "<=": # for numerical values
        if data[header] <= float(value):
            answer = subtree[question][0] #yesAnswer   
        else:
            answer = subtree[question][1] #noAnswer
    else:
    #if operator == "!=": # for nominal values
        if data[header] == value:
            answer = subtree[question][0] #yesAnswer
        else:
            answer = subtree[question][1] #noAnswer


    if isinstance(answer, str): #base 
        return answer # when answer is found
    
    else: # recursion
        subtree = answer
        return findAnswer(subtree, data)      
#findAnswer(tree, TrainSet.iloc[1])


# calculates TP, TN, FP, FN
def analyzeResults(subtree, data):
    TP, TN, FP, FN = 0, 0, 0, 0
    for dataQuestion in range(len(data)):
        prediction = findAnswer(subtree, data.iloc[dataQuestion])
      
        if data.iloc[dataQuestion][-1] == prediction:
            if data.iloc[dataQuestion][-1] == "good": # data with good class and correct prediction
                TP += 1 
            else: # data with bad class and correct prediction
                TN += 1
        if data.iloc[dataQuestion][-1] != prediction:
            if data.iloc[dataQuestion][-1] == "good": # data with good class and wrong prediction
                FP += 1
            else: # data with bad class and wrong prediction
                FN += 1
            
    return TP, TN, FP, FN


# prints results
def showResults(TP, TN, FP, FN):
    Accuracy = (TP + TN) / (TP + FN +TN + FP)
    TPrate = TP / (TP + FN)
    TNrate = TN / (TN + FP)
    Precision = TP / (TP + FP)
    FScore = 2 * (Precision * TPrate) / (Precision + TPrate)

    print("Accuracy: ", Accuracy)
    print("TPrate: ", TPrate)
    print("TNrate: ", TNrate)
    print("Precision: ", Precision)
    print("FScore: ", FScore)
    print("Total number of TP: ", TP)
    print("Total number of TN: ", TN)

    return

