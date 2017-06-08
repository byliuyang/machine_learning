from enum import Enum

import numpy as np


class Value(Enum):
    pass


class DecisionTree:
    class TreeNode:
        def __init__(self, label=None, values=None, children=[]):
            self.label = label
            self.values = values
            self.children = children

    def __init__(self):
        self._root = None

    def entropy(self, values):
        unique_values, counts = np.unique(values, return_counts=True)
        probabilities = [count / len(values) for count in counts]
        return np.sum([-probability * np.math.log(probability, 2) for probability in probabilities])

    def partition(self, attribute, classifications, values=None):
        if values is None:
            values = np.unique(attribute)
        return [np.array(classifications)[(np.array(attribute) == value).nonzero()[0]] for value in values]

    def information_gain(self, attribute, classifications):
        original_entropy = self.entropy(classifications)

        new_entropies = [len(values) / len(attribute) * self.entropy(values) for values in
                         self.partition(attribute, classifications)]
        new_weighted_entropy = np.sum(new_entropies)
        return original_entropy - new_weighted_entropy

    def is_pure(self, classifications):
        return len(set(classifications)) == 1

    def no_more_attribute(self, attributes):
        return len(attributes) == 0

    def slice(self, original_list, index):
        return original_list[:index] + original_list[index + 1:]

    def build_tree(self, attributes, attribute_labels, classifications, class_label):
        if self.is_pure(classifications):
            return self.TreeNode(class_label, [set(classifications).pop()])
        elif self.no_more_attribute(attributes):
            return self.TreeNode(class_label, np.unique(classifications).tolist())

        information_gains = [self.information_gain(attribute, classifications) for attribute in attributes]
        best_attribute_index = np.argmax(information_gains)
        best_attribute_label = attribute_labels[best_attribute_index]
        best_attribute = attributes[best_attribute_index]

        attributes = self.slice(attributes, best_attribute_index)
        attribute_labels = self.slice(attribute_labels, best_attribute_index)

        best_attribute_values = np.unique(best_attribute)
        classification_subsets = [subset.tolist() for subset in
                                  self.partition(best_attribute, classifications, best_attribute_values)]
        attributes_subsets = np.transpose(
            [[subset.tolist() for subset in self.partition(best_attribute, attribute, best_attribute_values)] for
             attribute in attributes]).tolist()

        subsets = zip(attributes_subsets, classification_subsets)
        best_attribute_values = [best_attribute_label(value) for value in best_attribute_values]
        children = [child for child in [self.build_tree(subset[0], attribute_labels, subset[1], class_label) for subset in subsets] if child is not None]
        return self.TreeNode(best_attribute_label, best_attribute_values, children)

    def fit(self, attributes, attribute_labels, classifications, class_label):
        attributes = np.transpose(attributes).tolist()
        self._root = self.build_tree(attributes, attribute_labels, classifications, class_label)

    def __predict(self, node, attributes, attribute_labels, class_label):
        if len(attributes) == 0 or node.label is class_label:
            return node.values[0]

        attribute_index = np.nonzero([node.label == label for label in attribute_labels])[0][0]
        attribute = attributes[attribute_index]

        value_index = (np.array([value.value for value in node.values]) == attribute).nonzero()[0][0]

        return self.__predict(node.children[value_index], self.slice(attributes, attribute_index), self.slice(attribute_labels, attribute_index), class_label)

    def predict(self, attributes, attribute_labels, class_label):
        if self._root is None:
            return None

        return [self.__predict(self._root, attribute, attribute_labels, class_label) for attribute in attributes]

    def accuracy(self, attributes, expected_class, attribute_labels, class_label):
        predicted_class = self.predict(attributes, attribute_labels, class_label)
        values, counts = np.unique([tuple[0] == tuple[1] for tuple in zip(expected_class, predicted_class)], return_counts=True)
        correct_count = counts[np.nonzero(values)[0]]
        return correct_count / len(expected_class)