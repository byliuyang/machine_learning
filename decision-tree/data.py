import numpy as np
from id3 import Value


class Outlook(Value):
    SUNNY = 5
    OVERCAST = 1
    RAIN = 3


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

x = np.transpose([ x_temperature, x_humidity, x_wind, x_outlook]).tolist()
attribute_labels = [Temperature, Humidity, Wind, Outlook]

y = [value.value for value in
     [Play.NO, Play.NO, Play.YES, Play.YES, Play.YES, Play.NO, Play.YES, Play.NO, Play.YES, Play.YES, Play.YES,
      Play.YES, Play.YES, Play.NO]]

class_label = Play