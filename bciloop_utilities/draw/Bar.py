#!/usr/bin/python
import cv2
import numpy

BAR_THICK = 3
BAR_COLOR_LEFT = (128, 0, 128)
BAR_COLOR_RIGHT = (50, 205, 50)
BAR_COLOR_CENTER = (255, 140, 0)
BAR_COLOR_VERY_LEFT = (75, 0, 130)
BAR_COLOR_VERY_RIGHT = (0, 128, 0)


# Bar class
class Bar:
    def __init__(self, window, idx, name=None):
        self.name = name
        self.bar_width = numpy.int32(window.width / 8)
        self.bar_height = numpy.int32(window.height / 4)
        bar_height_start = self.bar_height
        bar_height_stop = 2 * self.bar_height
        type = self.bar_type(idx)
        if type is 'left':
            self.bar_start = (2 * self.bar_width, bar_height_start)
            self.bar_stop = (3 * self.bar_width, bar_height_stop)
            self.color = BAR_COLOR_LEFT
        elif type is 'right':
            self.bar_start = (5 * self.bar_width, bar_height_start)
            self.bar_stop = (6 * self.bar_width, bar_height_stop)
            self.color = BAR_COLOR_RIGHT
        elif type is 'center':
            self.bar_start = (3 * self.bar_width + numpy.int32(self.bar_width / 2), bar_height_start)
            self.bar_stop = (4 * self.bar_width + numpy.int32(self.bar_width / 2), bar_height_stop)
            self.color = BAR_COLOR_CENTER
        elif type is 'very-left':
            self.bar_start = (numpy.int32(self.bar_width / 2), bar_height_start)
            self.bar_stop = (self.bar_width + numpy.int32(self.bar_width / 2), bar_height_stop)
            self.color = BAR_COLOR_VERY_LEFT
        elif type is 'very-right':
            self.bar_start = (6 * self.bar_width + numpy.int32(self.bar_width / 2), bar_height_start)
            self.bar_stop = (7 * self.bar_width + numpy.int32(self.bar_width / 2), bar_height_stop)
            self.color = BAR_COLOR_VERY_RIGHT
        else:
            print("Invalid bar type")
            self.bar_start = (0, 0)
            self.bar_stop = (0, 0)
            self.color = (0, 0, 0)
        self.alpha = 0.5
        self.value = 0.0

    def bar_type(self, i):
        switcher = {
            0: 'left',
            1: 'right',
            2: 'center',
            3: 'very-left',
            4: 'very-right'
        }
        return switcher.get(i, 'Invalid number of bars')

    def set_value(self, value):
        self.value = value

    def set_alpha(self, alpha):
        self.alpha = alpha

    def draw(self, canvas):
        bar_fill_start = (self.bar_start[0], int(self.bar_stop[1] - self.value * self.bar_height))
        tmp = canvas.copy()
        tmp = cv2.rectangle(tmp, bar_fill_start, self.bar_stop, self.color, -1)  # Fill feedback bars
        cv2.addWeighted(tmp, self.alpha, canvas, 1 - self.alpha, 0, canvas)
        canvas = cv2.rectangle(canvas, self.bar_start, self.bar_stop, (255, 255, 255), BAR_THICK)  # Draw feedback bars
        canvas = cv2.putText(canvas, self.name, (self.bar_stop[0] - self.bar_width,
                                                 self.bar_stop[1] + numpy.int32(self.bar_stop[1] / 8)),
                             20, 1, (255, 255, 255), 2, cv2.LINE_AA)    # Add text
        return canvas

