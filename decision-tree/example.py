from data import x, y, class_label, attribute_labels
from id3 import DecisionTree

decisionTree = DecisionTree()

print("Input:", x)
decisionTree.fit(x, attribute_labels, y, class_label)
print("Predicted class:", decisionTree.predict(x, attribute_labels, class_label))

test_y = [val for val in y]
test_y[5] = 1
test_y[8] = 0
print("Accuracy: ", decisionTree.accuracy(x, test_y, attribute_labels, class_label))