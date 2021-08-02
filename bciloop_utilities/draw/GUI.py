#!/usr/bin/python
import cv2
import numpy
from bciloop_utilities.draw.Window import Window
from bciloop_utilities.draw.Bar import Bar


# SMRGUI class
class SMRGUI:
    def __init__(self, win_height, win_width, win_scale):
        cv2.namedWindow('canvas', cv2.WINDOW_NORMAL)
        cv2.moveWindow('canvas', 0, 0)
        self.window = Window(win_height, win_width, win_scale)
        self.canvas = numpy.zeros((win_height, win_width, 3), numpy.uint8)
        self.bars = []

    def init_canvas(self):
        self.canvas = numpy.zeros((self.window.height, self.window.width, 3), numpy.uint8)

    def init_bars(self, num_classes, class_names=None):
        del self.bars[:]
        if class_names is not None:
            for i in range(num_classes):
                self.bars.append(Bar(self.window, i, class_names[i]))
        else:
            for i in range(num_classes):
                self.bars.append(Bar(self.window, i))

    def set_value_bars(self, value, idx):
        self.bars[idx].set_value(value)
        self.draw()

    def set_alpha_bars(self, alpha, idx):
        self.bars[idx].set_alpha(alpha)
        self.draw()

    def reset_bars(self):
        for bar in self.bars:
            bar.set_value(0.0)
            bar.set_alpha(0.5)
        self.draw()

    def normalize_probabilities(self, value, max, min):
        nvalue = ((1.0 - 0.0) * (value - min)) / (max - min) + 0.0
        nvalue = 1.0 if nvalue >= 1.0 else nvalue
        nvalue = 0.0 if nvalue <= 0.0 else nvalue
        return nvalue

    def draw(self):
        self.init_canvas()
        for bar in self.bars:
            self.canvas = bar.draw(self.canvas)

        canvas = cv2.resize(self.canvas, (self.window.width * self.window.scale, self.window.height * self.window.scale))
        cv2.imshow('canvas', canvas)
        cv2.waitKey(1)
