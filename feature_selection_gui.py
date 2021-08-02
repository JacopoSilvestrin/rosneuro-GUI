import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from ttkthemes import ThemedStyle
import numpy as np
import seaborn as sns
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def create_figure(matrix):
    # plot the data
    figure = Figure(figsize=(4, 3))
    ax = figure.add_axes([0,0,1,1])
    sns.heatmap(matrix, square=False, cbar=False, annot=True, ax=ax)
    return figure

class FeatureSelectionGUI:

    def select_feature(self, event):
        x = event.x
        y = event.y
        width = event.widget.winfo_reqwidth()
        height = event.widget.winfo_reqheight()
        cell_width = width / self.ncolumns
        cell_height = height / self.nrows
        row = int(y // cell_height)
        column = int(x // cell_width)

        if self.mask[row,column] == 1:
            self.mask[row,column] = 0
        else:
            self.mask[row,column] = 1

        self.mask_figure.clf()
        #self.mask_figure = Figure(figsize=(4, 3))
        mask_ax = self.mask_figure.add_axes([0,0,1,1])
        sns.heatmap(self.mask, square=False, cbar=False, annot=True, ax=mask_ax)
        self.maskCanvas.draw()

    def automatic_selection(self, threshold):

        if self.autoVar == 0:
            return

        threshold = float(threshold)
        for row in range(self.nrows):
            for col in range(self.ncolumns):
                if self.mean_fisherscore[row,col] >= threshold:
                    self.mask[row,col] = 1
                else:
                    self.mask[row,col] = 0

        self.mask_figure.clf()
        mask_ax = self.mask_figure.add_axes([0,0,1,1])
        sns.heatmap(self.mask, square=False, cbar=False, annot=True, ax=mask_ax)
        self.maskCanvas.draw()

    def quit(self):
       self.root.destroy()

    def __init__(self, single_file_fisherscore, mean_fisherscore):

        #master.configure(background='grey')
        self.root = tk.Tk()
        self.root.title("Feature Selection GUI")

        #matrix cspdim x nbands
        self.single_file_fisherscore = single_file_fisherscore
        self.mean_fisherscore = mean_fisherscore

        rows = []
        columns = []

        for i in range(np.shape(self.mean_fisherscore)[0]):
            rows.append(i+1)
        for i in range(np.shape(self.mean_fisherscore)[1]):
            columns.append(i+1)

        # GET PARAMETERS
        self.nrows  = len(rows)
        self.ncolumns = len(columns)
        nfiles = np.shape(self.single_file_fisherscore)[2]
        rows = tuple(rows)
        columns = tuple(columns)
        #create mask matrix
        self.mask = np.zeros((self.nrows,self.ncolumns))

        #--------root size
        #master.geometry("900x500")
        self.style = ThemedStyle()
        self.style.set_theme('arc')

        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)


        #upper section
        self.upperFrame = ttk.LabelFrame(self.root, text="Single file feature scores:")
        self.upperFrame.grid(row=1, column=0, columnspan=3, sticky=tk.EW)

        self.upperFrame.columnconfigure(0, weight=1)
        self.upperFrame.columnconfigure(1, weight=1)
        self.upperFrame.columnconfigure(2, weight=1)


        #self.upperFrame.columnconfigure(1, weight=1)
        if(nfiles > 2):
            firstFile = np.reshape(self.single_file_fisherscore[:, :, 0], (self.nrows, self.ncolumns))
            secondFile = np.reshape(self.single_file_fisherscore[:, :, 1], (self.nrows, self.ncolumns))
            thirdFile = np.reshape(self.single_file_fisherscore[:, :, 2], (self.nrows, self.ncolumns))

            self.first_figure = Figure(figsize=(4, 3))
            first_ax = self.first_figure.add_axes([0,0,1,1])
            sns.heatmap(firstFile, square=False, cbar=False, annot=True, ax=first_ax)
            #first_figure = create_figure(firstFile)
            self.firstCanvas =  FigureCanvasTkAgg(self.first_figure, master=self.upperFrame)
            self.firstCanvas.get_tk_widget().grid(row = 0, column=0, padx = (15,15), pady=(15,15), sticky=tk.N+tk.S+tk.W+tk.E)
            self.firstCanvas.draw()
            self.firstCanvas.get_tk_widget().bind("<Button-1>", self.select_feature)

            self.second_figure = Figure(figsize=(4, 3))
            second_ax = self.second_figure.add_axes([0,0,1,1])
            sns.heatmap(secondFile, square=False, cbar=False, annot=True, ax=second_ax)
            #second_figure = create_figure(secondFile)
            self.secondCanvas =  FigureCanvasTkAgg(self.second_figure, master=self.upperFrame)
            self.secondCanvas.get_tk_widget().grid(row = 0, column=1, padx = (15,15), pady=(15,15), sticky=tk.N+tk.S+tk.W+tk.E)
            self.secondCanvas.draw()
            self.secondCanvas.get_tk_widget().bind("<Button-1>", self.select_feature)

            self.third_figure = Figure(figsize=(4, 3))
            third_ax = self.third_figure.add_axes([0,0,1,1])
            sns.heatmap(thirdFile, square=False, cbar=False, annot=True, ax=third_ax)
            #third_figure = create_figure(thirdFile)
            self.thirdCanvas =  FigureCanvasTkAgg(self.third_figure, master=self.upperFrame)
            self.thirdCanvas.get_tk_widget().grid(row = 0, column=2, padx = (15,15), pady=(15,15), sticky=tk.N+tk.S+tk.W+tk.E)
            self.thirdCanvas.draw()
            self.thirdCanvas.get_tk_widget().bind("<Button-1>", self.select_feature)

        if(nfiles == 1):
            firstFile = np.reshape(self.single_file_fisherscore[:, :, 0], (self.nrows, self.ncolumns))

            self.first_figure = Figure(figsize=(4, 3))
            first_ax = self.first_figure.add_axes([0,0,1,1])
            sns.heatmap(firstFile, square=False, cbar=False, annot=True, ax=first_ax)
            self.firstCanvas =  FigureCanvasTkAgg(self.first_figure, master=self.upperFrame)
            self.firstCanvas.get_tk_widget().grid(row = 0, column=0, padx = (15,15), pady=(15,15), sticky=tk.N+tk.S+tk.W+tk.E)
            self.firstCanvas.draw()
            self.firstCanvas.get_tk_widget().bind("<Button-1>", self.select_feature)

        if(nfiles == 2):
            firstFile = np.reshape(self.single_file_fisherscore[:, :, 0], (self.nrows, self.ncolumns))
            secondFile = np.reshape(self.single_file_fisherscore[:, :, 1], (self.nrows, self.ncolumns))

            self.first_figure = Figure(figsize=(4, 3))
            first_ax = self.first_figure.add_axes([0,0,1,1])
            sns.heatmap(firstFile, square=False, cbar=False, annot=True, ax=first_ax)
            self.firstCanvas =  FigureCanvasTkAgg(self.first_figure, master=self.upperFrame)
            self.firstCanvas.get_tk_widget().grid(row = 0, column=0, padx = (15,15), pady=(15,15), sticky=tk.N+tk.S+tk.W+tk.E)
            self.firstCanvas.draw()
            self.firstCanvas.get_tk_widget().bind("<Button-1>", self.select_feature)

            self.second_figure = Figure(figsize=(4, 3))
            second_ax = self.second_figure.add_axes([0,0,1,1])
            sns.heatmap(secondFile, square=False, cbar=False, annot=True, ax=second_ax)
            self.secondCanvas =  FigureCanvasTkAgg(self.second_figure, master=self.upperFrame)
            self.secondCanvas.get_tk_widget().grid(row = 0, column=1, padx = (15,15), pady=(15,15), sticky=tk.N+tk.S+tk.W+tk.E)
            self.secondCanvas.draw()
            self.secondCanvas.get_tk_widget().bind("<Button-1>", self.select_feature)

        #LOWER SECTION
        # lower left frame
        self.lowerLeftFrame = ttk.LabelFrame(self.upperFrame, text="Mean feature score")
        self.lowerLeftFrame.grid(row=1, column=0, sticky=tk.N+tk.S+tk.W+tk.E)

        self.mean_figure = Figure(figsize=(4, 3))
        mean_ax = self.mean_figure.add_axes([0,0,1,1])
        sns.heatmap(self.mean_fisherscore, square=False, cbar=False, annot=True, ax=mean_ax)
        #mean_figure = create_figure(self.mean_fisherscore)
        self.meanCanvas = FigureCanvasTkAgg(self.mean_figure, master=self.lowerLeftFrame)
        self.meanCanvas.get_tk_widget().grid(row=0, column=0, padx = (15,15), pady=(15,15), sticky=tk.E+tk.W+tk.N+tk.S)
        self.meanCanvas.draw()
        self.meanCanvas.get_tk_widget().bind("<Button-1>", self.select_feature)

        #lower central frame
        self.lowerCenterFrame = ttk.LabelFrame(self.upperFrame, text="Selected features")
        self.lowerCenterFrame.grid(row=1, column=1, sticky=tk.N+tk.S+tk.W+tk.E)

        self.mask_figure = Figure(figsize=(4, 3))
        mask_ax = self.mask_figure.add_axes([0,0,1,1])
        sns.heatmap(self.mask, square=False, cbar=False, annot=False, ax=mask_ax)
        #self.mask_figure = create_figure(self.mask)
        self.maskCanvas = FigureCanvasTkAgg(self.mask_figure, master=self.lowerCenterFrame)
        self.maskCanvas.get_tk_widget().grid(row=0, column=2, padx = (15,15), pady=(15,15), sticky=tk.E+tk.W+tk.N+tk.S)
        self.maskCanvas.draw()
        self.maskCanvas.get_tk_widget().bind("<Button-1>", self.select_feature)

        #lower right frame
        self.lowerRightFrame = ttk.LabelFrame(self.upperFrame, text="Selection settings")
        self.lowerRightFrame.grid(row=1, column=2, sticky=tk.N+tk.S+tk.W+tk.E)

        self.selLabel = ttk.Label(self.lowerRightFrame, text="Selection Threshold")
        self.selLabel.grid(row=0, column=1, pady=25, padx= 10)

        self.selFrame = ttk.LabelFrame(self.lowerRightFrame, text="Feature selection Mode")
        self.selFrame.grid(row=1, column=0, pady=25, padx=15)

        self.autoVar = tk.IntVar()
        self.autoCheckbutton = ttk.Checkbutton(self.selFrame, text="Automatic", variable=self.autoVar, onvalue=1, offvalue=0)
        self.autoCheckbutton.grid(row=0, column=0, sticky=tk.W)

        self.thresholdVar = tk.DoubleVar()
        maxFisher = np.amax(self.single_file_fisherscore)
        minFisher = np.amin(self.single_file_fisherscore)
        self.thresholdVar.set(0.5*(maxFisher-minFisher))

        self.scaleLabel = ttk.Label(self.selFrame, textvariable = self.thresholdVar)
        self.scaleLabel.grid(row=1, column=0, sticky=tk.E, pady = 10)

        self.trainButton = ttk.Button(self.lowerRightFrame, text="Train Classifier", command=self.quit)
        self.trainButton.grid(row=2, column=0, pady=25, ipadx=10, ipady=10)

        self.threshScale = ttk.Scale(self.lowerRightFrame, from_=maxFisher, to=minFisher, orient=tk.VERTICAL, variable = self.thresholdVar, command= self.automatic_selection)
        self.threshScale.grid(row=1, column=1, rowspan=2, sticky=tk.N+tk.S, padx=10)
