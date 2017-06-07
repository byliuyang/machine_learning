from enum import Enum
import numpy as np


class Value(Enum):
    pass


class Outlook(Value):
    SUNNY = 0
    OVERCAST = 1
    RAIN = 2


class Temperature(Value):
    HOT = 0
    MILD = 1
    COOL = 2


class Humidity(Value):
    HIGH = 0
    NORMAL = 1


class Wind(Value):
    WEAK = 0
    STRONG = 1


class Play(Value):
    NO = 0
    YES = 1


def to_values(enums):
    return [value.value for value in enums]


x_outlook = to_values([Outlook.SUNNY, Outlook.SUNNY, Outlook.OVERCAST, Outlook.RAIN, Outlook.RAIN, Outlook.RAIN,
                       Outlook.OVERCAST,
                       Outlook.SUNNY, Outlook.SUNNY, Outlook.RAIN, Outlook.SUNNY, Outlook.OVERCAST, Outlook.OVERCAST,
                       Outlook.RAIN])

x_temperature = to_values([Temperature.HOT, Temperature.HOT, Temperature.HOT, Temperature.MILD, Temperature.COOL,
                           Temperature.COOL,
                           Temperature.COOL, Temperature.MILD, Temperature.COOL, Temperature.MILD, Temperature.MILD,
                           Temperature.MILD, Temperature.HOT, Temperature.MILD])

x_humidity = to_values([Humidity.HIGH, Humidity.HIGH, Humidity.HIGH, Humidity.HIGH, Humidity.NORMAL, Humidity.NORMAL,
                        Humidity.NORMAL, Humidity.HIGH, Humidity.NORMAL, Humidity.NORMAL, Humidity.NORMAL,
                        Humidity.HIGH,
                        Humidity.NORMAL, Humidity.HIGH])

x_wind = to_values(
    [Wind.WEAK, Wind.STRONG, Wind.WEAK, Wind.WEAK, Wind.WEAK, Wind.STRONG, Wind.STRONG, Wind.WEAK, Wind.WEAK,
     Wind.WEAK, Wind.STRONG, Wind.STRONG, Wind.WEAK, Wind.STRONG])

x = [x_outlook, x_temperature, x_humidity, x_wind]
attribute_labels = [Outlook, Temperature, Humidity, Wind]


y = [value.value for value in
     [Play.NO, Play.NO, Play.YES, Play.YES, Play.YES, Play.NO, Play.YES, Play.NO, Play.YES, Play.YES, Play.YES,
      Play.YES, Play.YES, Play.NO]]


class DecisionTree:
    class TreeNode:
        def __init__(self, attribute_label, values, children):
            self.attribute_label = attribute_label
            self.values = values
            self.children = children

    def __init__(self):
        self._root = None

    def entropy(self, values):
        unique_values, counts = np.unique(values, return_counts=True)
        probabilities = [count / len(values) for count in counts]
        return np.sum([-probability * np.math.log(probability, 2) for probability in probabilities])

    def partition(self, attribute, classifications):
        return [np.array(classifications)[(np.array(attribute) == value).nonzero()[0]] for value in np.unique(attribute)]

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

    def build_tree(self, attributes, attribute_labels, classifications):
        if self.is_pure(classifications) or self.no_more_attribute(attributes):
            return None

        information_gains = [self.information_gain(attribute, classifications) for attribute in attributes]
        best_attribute_index = np.argmax(information_gains)
        best_attribute_label = attribute_labels[best_attribute_index]
        best_attribute = attributes[best_attribute_index]


        attributes = attributes[:best_attribute_index] + attributes[best_attribute_index + 1:]
        attribute_labels = attribute_labels[:best_attribute_index] + attribute_labels[
                                                                                 best_attribute_index + 1:]
        classification_subsets = [subset.tolist() for subset in self.partition(best_attribute, classifications)]
        attributes_subsets = np.transpose([[subset.tolist() for subset in self.partition(best_attribute, attribute)] for attribute in attributes]).tolist()

        subsets = zip(attributes_subsets, classification_subsets)
        return self.TreeNode(best_attribute_label, [best_attribute_label(value) for value in np.unique(best_attribute)],
                             [self.build_tree(subset[0], attribute_labels, subset[1]) for subset in subsets])

    def fit(self, attributes, attribute_labels, classifications):
        self._root = self.build_tree(attributes, attribute_labels, classifications)


decisionTree = DecisionTree()
decisionTree.fit(x, attribute_labels, y)

decisionTree._root