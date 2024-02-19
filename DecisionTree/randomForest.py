from Tree import *

# bootsrapping
def bootstrap(data, bootstrapCount): # data = training set
    index = np.random.randint(low=0, high=len(data), size=bootstrapCount)
    bootstrappedData = data.iloc[index]
    return bootstrappedData

# creating random forest
def createRandomForest(TrainSet, TestSet, bootstrapCount, depthLimit):
    TP, TN, FP, FN = 0, 0, 0, 0

    bootstrappedData = bootstrap(TrainSet, bootstrapCount)
    trees = []
    for i in bootstrappedData:
        tree = createDecisionTree(data=bootstrappedData, depthLimit=depthLimit)
        trees.append(tree)

    # asks questions
    for question in range(len(TestSet)):
        answers = []
        for tree in trees:
            answers.append(findAnswer(tree, TestSet.iloc[question]))
        
        prediction = max(set(answers), key=answers.count)

        # analyzes and calculates TP, TN, FP, FN
        if TestSet.iloc[question][-1] == prediction:
            if TestSet.iloc[question][-1] == "good": # data with good class and correct prediction
                TP += 1 
            else: # data with bad class and correct prediction
                TN += 1
        if TestSet.iloc[question][-1] != prediction:
            if TestSet.iloc[question][-1] == "good": # data with good class and wrong prediction
                FP += 1
            else: # data with bad class and wrong prediction
                FN += 1
            
    return TP, TN, FP, FN
#createRandomForest(TrainSet, TestSet, 4, 7)
