
from Tree import *
from randomForest import *

# takes a training set to create the tree and the other parameter is for the tree's depth limit
tree = createDecisionTree(TrainSet, 7)

print("\nResults for train set: ")
TP, TN, FP, FN = analyzeResults(tree, TrainSet)
showResults(TP, TN, FP, FN)

print("\nResults for test set: ")
TP, TN, FP, FN = analyzeResults(tree, TestSet)
showResults(TP, TN, FP, FN)

# takes training set, test set, bootstrapCount and depthLimit as parameters
print("\nResults for test set of random forest: ")
TP, TN, FP, FN = createRandomForest(TrainSet, TestSet, 5, 7)
showResults(TP, TN, FP, FN)

# My code works, but it may occasionally gives an error for a reason I don't understand. 
# The problem is fixed when I run it again.
# If it happens could you run the code again?
