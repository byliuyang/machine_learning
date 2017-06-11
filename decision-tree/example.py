from data import x, y, class_label, attribute_labels
from id3 import ID3DecisionTree

id3decisionTree = ID3DecisionTree()

print("Input:", x)
id3decisionTree.fit(x, attribute_labels, y, class_label)
print("Predicted class:", id3decisionTree.predict(x, attribute_labels, class_label))

test_y = [val for val in y]
test_y[5] = 1
test_y[8] = 0
print("Accuracy: ", id3decisionTree.accuracy(x, test_y, attribute_labels, class_label))