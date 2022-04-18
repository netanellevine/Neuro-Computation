class Point:
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._value = 1 if y > 1 else -1

    def getX(self):
        return self._x

    def getY(self):
        return self._y

    def getValue(self):
        return self._value

    def setX(self, x):
        self._x = x

    def setY(self, y):
        self._y = y

    def setValue(self, value):
        self._value = value
