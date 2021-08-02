import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from ttkthemes import ThemedStyle
from xml.dom import minidom



def get_param_from_xml(doc, name):
    item = doc.getElementsByTagName(name)
    return item[0].firstChild.data

class EntropyTrainGUI:

    def select_gdf_file(self):
        filetypes = (
            ('gdf files', '*.gdf'),
            ('All files', '*.*')
        )

        filename = fd.askopenfilename(
            title='Open the data file',
            initialdir='/home',
            filetypes=filetypes)

        if not filename:
            return

        #self.walk_path = filename
        self.loadListbox.insert(tk.END, filename)
        self.files.append(filename)


    def delete_file(self):
        self.loadListbox.delete(tk.ANCHOR)
        self.files.remove(str(self.loadListbox.get(ACTIVE)))

    def select_lap_file(self):
        filetypes = (
            ('numpy files', '*.npy'),
            ('All files', '*.*')
        )

        filename = fd.askopenfilename(
            title='Open laplacian mask file',
            initialdir='/home',
            filetypes=filetypes)

        if not filename:
            return

        self.lap_path.set(filename)

    def insert_class(self, label, code, insert):
        #if we have to add the class
        if(insert):
            #if it is the first class
            if(label == 0):
                self.first_class_codes.append(code)
            else:
                self.second_class_codes.append(code)
        else:
            if(label == 0):
                self.first_class_codes.remove(code)
            else:
                self.second_class_codes.remove(code)

    def add_filter(self):
        order = self.filter_order.get()
        low_f = self.filter_lowf.get()
        high_f = self.filter_highf.get()
        self.filterTV.insert(parent='', index=tk.END, iid=self.filterIid, values=(order, low_f, high_f))
        self.filter_order_list.append(order)
        self.filter_lowf_list.append(low_f)
        self.filter_highf_list.append(high_f)
        self.filterIid = self.filterIid + 1


    def remove_filter(self):
        selected_items = self.filterTV.selection() ## get selected items

        for item in selected_items:
            values = self.filterTV.item(item)['values']
            order = int(values[0])
            low_f = float(values[1])
            high_f = float(values[2])
            self.filter_order_list.remove(order)
            self.filter_lowf_list.remove(low_f)
            self.filter_highf_list.remove(high_f)

            self.filterTV.delete(item)

    def remove_all_filters(self):
        for item in self.filterTV.get_children():
            self.filterTV.delete(item)
        self.filter_order_list = []
        self.filter_lowf_list = []
        self.filter_highf_list = []
        self.filterIid = 0

    def reset_values(self):
        self.window_length.set(1.5)
        self.window_shift.set(0.125)
        self.srate.set(512)
        self.filter_order.set(4)
        self.filter_lowf.set(16)
        self.filter_highf.set(30)
        self.nbins.set(32)
        self.cspdimm.set(8)
        self.alpha.set(0.9)
        self.threshold.set(0.5)
        self.rejection.set(0.5)
        self.begin.set(0)
        self.nfold.set(5)
        self.remove_all_filters()

    def quit(self):
       self.root.destroy()

    def __init__(self):

        self.root = tk.Tk()

        #parse settings file
        xml_settings = minidom.parse('train_gui_settings.xml')

        load_csp = int(get_param_from_xml(xml_settings, 'csp'))
        load_car = int(get_param_from_xml(xml_settings, 'car'))
        load_laplacian = int(get_param_from_xml(xml_settings, 'lap'))
        load_temporal = int(get_param_from_xml(xml_settings, 'temporal'))
        load_feature_selection = str(get_param_from_xml(xml_settings, 'feature_score'))
        load_entropy = str(get_param_from_xml(xml_settings, 'entropy'))
        load_integrator = int(get_param_from_xml(xml_settings, 'integrator'))

        # variables
        self.window_length = tk.DoubleVar()
        self.window_shift = tk.DoubleVar()
        self.srate = tk.IntVar()
        self.filter_order = tk.IntVar()
        self.filter_lowf = tk.DoubleVar()
        self.filter_highf = tk.DoubleVar()
        self.filterIid = 0
        self.filter_order_list = []
        self.filter_lowf_list = []
        self.filter_highf_list = []
        self.nbins = tk.IntVar()
        self.cspdimm = tk.IntVar()
        self.alpha = tk.DoubleVar()
        self.threshold =tk.DoubleVar()
        self.rejection = tk.DoubleVar()
        self.begin = tk.DoubleVar()
        self.load_data = tk.IntVar()
        self.save_classifier = tk.IntVar()
        self.className = tk.StringVar()

        self.files = []
        self.first_class_codes = []
        self.second_class_codes = []
        self.laplacian = tk.IntVar()
        self.lap_path = tk.StringVar()
        self.car_filter = tk.IntVar()


        #SET THE VARIABLES TO DEFAULT VALUES

        self.lap_path.set("No file selected.")
        self.window_length.set(1.5)
        self.window_shift.set(0.125)
        self.srate.set(512)
        self.filter_order.set(4)
        self.filter_lowf.set(16)
        self.filter_highf.set(30)
        self.nbins.set(32)
        self.cspdimm.set(8)
        self.alpha.set(0.9)
        self.threshold.set(0.5)
        self.rejection.set(0.5)
        self.begin.set(0)


        self.root.title("Classifier Training")

        self.style = ThemedStyle(self.root)
        self.style.set_theme('arc')

        # root configuration
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=1)
        self.root.columnconfigure(3, weight=1)

        # GENERAL FRAME STRUCTURE

        #upper frame
        self.settingsFrame = ttk.LabelFrame(self.root, text="Settings")
        self.settingsFrame.grid(row=0,column=0, sticky=tk.N+tk.S+tk.W+tk.E, columnspan=4)
        self.settingsFrame.columnconfigure(0, weight=1)
        self.settingsFrame.columnconfigure(1, weight=1)

        #lower left
        self.classSelFrame = ttk.LabelFrame(self.root, text="Load gdf")
        self.classSelFrame.grid(row=1,column=0, sticky=tk.N+tk.S+tk.W+tk.E, columnspan=1, rowspan=2, pady=10)
        self.classSelFrame.columnconfigure(0, weight=1)
        self.classSelFrame.rowconfigure(0, weight=1)

        #lower right
        self.codeSelFrame = ttk.LabelFrame(self.root, text="Class Selection")
        self.codeSelFrame.grid(row=1, column=1, columnspan=3, sticky=tk.N+tk.S+tk.W+tk.E, pady=(10,0))
        self.guiOptionsFrame = ttk.LabelFrame(self.root, text="GUI Options")
        self.guiOptionsFrame.grid(row=2,column=1, columnspan=3, sticky=tk.N+tk.S+tk.W+tk.E, pady=(0,10))
        self.guiOptionsFrame.columnconfigure(0,weight=1)


        #UPPER FRAME

        #upper left
        self.leftFrame = ttk.Frame(self.settingsFrame)
        self.leftFrame.grid(row=0, column=0, sticky=tk.N+tk.S+tk.W+tk.E)
        self.leftFrame.rowconfigure(1, weight=1)

        self.rightFrame = ttk.Frame(self.settingsFrame)
        self.rightFrame.grid(row=0, column=1, sticky=tk.N+tk.S+tk.W+tk.E)
        self.rightFrame.rowconfigure(3, weight=1)

        self.bufferFrame = ttk.LabelFrame(self.leftFrame, text="Buffer Settings")
        self.bufferFrame.grid(row=0, column=0, sticky=tk.N+tk.S+tk.W+tk.E, padx=5, pady=5)
        self.bufferFrame.columnconfigure(0, weight=1)

        self.codeLabel = ttk.Label(self.bufferFrame, text="Sampling Rate:")
        self.codeLabel.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.codeEntry = ttk.Entry(self.bufferFrame, textvariable=self.srate)
        self.codeEntry.grid(row=0, column=1, sticky=tk.E, padx=5, pady=5)

        self.winSizeLabel = ttk.Label(self.bufferFrame, text="Window Lenght:")
        self.winSizeLabel.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.winSizeEntry = ttk.Entry(self.bufferFrame, textvariable=self.window_length)
        self.winSizeEntry.grid(row=1, column=1, sticky=tk.E, padx=5, pady=5)

        self.winShiftLabel = ttk.Label(self.bufferFrame, text="Window Shift:")
        self.winShiftLabel.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.winShiftEntry = ttk.Entry(self.bufferFrame, textvariable=self.window_shift)
        self.winShiftEntry.grid(row=2, column=1, sticky=tk.E, padx=5, pady=5)


        # filter settings frame
        self.filterFrame = ttk.LabelFrame(self.leftFrame, text="Filter Settings")
        self.filterFrame.grid(row=1, column=0, sticky=tk.N+tk.S+tk.W+tk.E, padx=5, pady=5)

        if (load_temporal):
            self.orderLabel = ttk.Label(self.filterFrame, text="Filter Order:")
            self.orderLabel.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
            self.orderEntry = ttk.Entry(self.filterFrame, textvariable=self.filter_order)
            self.orderEntry.grid(row=0, column=1, sticky=tk.E, padx=5, pady=5)

            self.lcLabel = ttk.Label(self.filterFrame, text="Lower cut-off frequency:")
            self.lcLabel.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
            self.lcEntry = ttk.Entry(self.filterFrame, textvariable=self.filter_lowf)
            self.lcEntry.grid(row=1, column=1, sticky=tk.E, padx=5, pady=5)

            self.hcLabel = ttk.Label(self.filterFrame, text="Higher cut-off frequency:")
            self.hcLabel.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
            self.hcEntry = ttk.Entry(self.filterFrame, textvariable=self.filter_highf)
            self.hcEntry.grid(row=2, column=1, sticky=tk.E, padx=5, pady=5)

            #table frame
            self.tableFrame = ttk.Frame(self.filterFrame)
            self.tableFrame.grid(row = 3, column=0, columnspan = 2)
            self.tableFrame.columnconfigure(0, weight=1)

            self.filterTV = ttk.Treeview(self.tableFrame, columns = (1,2,3), show='headings')
            self.filterTV.grid(row = 0, column=0)

            self.filterTV.heading(1, text="Order")
            self.filterTV.heading(2, text="Lower cut-off")
            self.filterTV.heading(3, text="Higher cut-off")

            self.filtersb = ttk.Scrollbar(self.tableFrame, orient=tk.VERTICAL)
            self.filtersb.grid(row=0, column=1, sticky=tk.N+tk.S)

            self.filterTV.config(yscrollcommand=self.filtersb.set)
            self.filtersb.config(command=self.filterTV.yview)

            self.buttonFrame = ttk.Frame(self.filterFrame)
            self.buttonFrame.grid(row=4, column=1, sticky=tk.W+tk.E+tk.N+tk.S, padx=5, pady=5)
            self.buttonFrame.columnconfigure(0, weight=1)

            self.delFilterButton = ttk.Button(self.buttonFrame, text="Delete Filter", command=self.remove_filter)
            self.delFilterButton.grid(row=0, column=0, sticky=tk.E)

            self.addFilterButton = ttk.Button(self.buttonFrame, text="Add Filter", command=self.add_filter)
            self.addFilterButton.grid(row=0, column=1, sticky=tk.E)



        # upper right framme
        self.classFrame = ttk.LabelFrame(self.rightFrame, text="Classification Settings")
        self.classFrame.grid(row=0, column=0, sticky=tk.N+tk.S+tk.W+tk.E, padx=5, pady=5)
        self.classFrame.columnconfigure(0, weight=1)

        if (load_entropy):
            self.binsLabel = ttk.Label(self.classFrame, text="Entropy Bins Number:")
            self.binsLabel.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
            self.binsEntry = ttk.Entry(self.classFrame, textvariable=self.nbins)
            self.binsEntry.grid(row=0, column=1, sticky=tk.E, padx=5, pady=5)

        if (load_csp):
            self.cspLabel = ttk.Label(self.classFrame, text="CSP Dimension:")
            self.cspLabel.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
            self.cspEntry = ttk.Entry(self.classFrame, textvariable=self.cspdimm)
            self.cspEntry.grid(row=1, column=1, sticky=tk.E, padx=5, pady=5)


        if (load_integrator):
            #preprocessing frame
            self.intFrame = ttk.LabelFrame(self.rightFrame, text="Integrator Settings")
            self.intFrame.grid(row=1, column=0, sticky=tk.N+tk.S+tk.W+tk.E, padx=5, pady=5)
            self.intFrame.columnconfigure(0, weight=1)

            self.alphaLabel = ttk.Label(self.intFrame, text="Alpha:")
            self.alphaLabel.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
            self.alphaEntry = ttk.Entry(self.intFrame, textvariable=self.alpha)
            self.alphaEntry.grid(row=0, column=1, sticky=tk.E, padx=5, pady=5)

            self.thresholdLabel = ttk.Label(self.intFrame, text="Integrator Threshold:")
            self.thresholdLabel.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
            self.thresholdEntry = ttk.Entry(self.intFrame, textvariable=self.threshold)
            self.thresholdEntry.grid(row=1, column=1, sticky=tk.E, padx=5, pady=5)

            self.rejLabel = ttk.Label(self.intFrame, text="Integrator Rejection:")
            self.rejLabel.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
            self.rejEntry = ttk.Entry(self.intFrame, textvariable=self.rejection)
            self.rejEntry.grid(row=2, column=1, sticky=tk.E, padx=5, pady=5)

            self.beginLabel = ttk.Label(self.intFrame, text="Integratore Begin:")
            self.beginLabel.grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
            self.beginEntry = ttk.Entry(self.intFrame, textvariable=self.begin)
            self.beginEntry.grid(row=3, column=1, sticky=tk.E, padx=5, pady=5)

        #training frame
        self.trainingFrame = ttk.LabelFrame(self.rightFrame, text="Training Settings")
        self.trainingFrame.grid(row=2, column=0, sticky=tk.N+tk.S+tk.W+tk.E, padx=5, pady=5)

        self.saveCheckbutton = ttk.Checkbutton(self.trainingFrame, text="Save Classifier", variable=self.save_classifier, onvalue=1, offvalue=0)
        self.saveCheckbutton.grid(row=0, column=0, sticky=tk.W, padx=5, pady=(5,5))

        self.classNameLabel = ttk.Label(self.trainingFrame, text="Classifier ID:")
        self.classNameLabel.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.classNameEntry = ttk.Entry(self.trainingFrame, textvariable=self.className)
        self.classNameEntry.grid(row=1, column=1, sticky=tk.E, padx=5, pady=5)

        self.preprocessingFrame = ttk.LabelFrame(self.rightFrame, text="Preprocessing Options")
        self.preprocessingFrame.grid(row=3, column=0, padx=10, pady=10, sticky=tk.N+tk.S+tk.W+tk.E)

        if (load_car and load_laplacian):
            self.laplacianCheckbutton = ttk.Checkbutton(self.preprocessingFrame, text="Laplacian", variable=self.laplacian, onvalue=1, offvalue=0)
            self.laplacianCheckbutton.grid(row=0, column=0, sticky=tk.W, padx=5, pady=(5,5))

            self.carCheckbutton = ttk.Checkbutton(self.preprocessingFrame, text="CAR", variable=self.car_filter, onvalue=1, offvalue=0)
            self.carCheckbutton.grid(row=0, column=1, sticky=tk.W, padx=5, pady=(5,5))

            self.lapButton = ttk.Button(self.preprocessingFrame, text="Select file", command=self.select_lap_file)
            self.lapButton.grid(row=1, column=0, sticky=tk.W, padx=5, pady=(5,5))

            self.lapLabel = ttk.Label(self.preprocessingFrame, textvariable = self.lap_path)
            self.lapLabel.grid(row=1, column=1, sticky=tk.E+tk.W, padx=5, pady=(5,5))


        self.resetButton = ttk.Button(self.rightFrame, text="Restore Default Settings", command=self.reset_values)
        self.resetButton.grid(row=4, column=0, sticky=tk.S+tk.W+tk.E, padx=15, pady=(30,30))

        #LOWER LEFT FRAME

        self.loadListbox = tk.Listbox(self.classSelFrame)
        self.loadListbox.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N+tk.S, columnspan= 2, padx=15, pady=(5,5))

        self.walkingButton = ttk.Button(self.classSelFrame, text="Load File", command=self.select_gdf_file)
        self.walkingButton.grid(row=1, column=1, sticky=tk.W+tk.E+tk.N, padx=5, pady=(5,20))

        self.standingButton = ttk.Button(self.classSelFrame, text="Delete", command=self.delete_file)
        self.standingButton.grid(row=1, column=0, sticky=tk.E, padx=5, pady=(5,20))


        #LOWER RIGHT FRAME

        #Class selection frame

        #self.codeSelFrame.columnconfigure(0, weight=1)
        #self.codeSelFrame.columnconfigure(1, weight=1)
        #first column
        self.firstClassLabel = ttk.Label(self.codeSelFrame, text="First Class Codes:")
        self.firstClassLabel.grid(row=0, column=0, sticky=tk.W+tk.E, padx=(30,15), pady=(5,5))

        self.firstNewVar = tk.IntVar()
        self.firstNewCodeEntry = ttk.Entry(self.codeSelFrame, textvariable=self.firstNewVar)
        self.firstNewCodeEntry.grid(row=1, column=0, sticky=tk.W+tk.E, padx=(30,5), pady=(5,5))
        self.firstNewCodeButton = ttk.Button(self.codeSelFrame, text="Add", command=lambda: self.insert_class(0, self.firstNewVar.get(),1))
        self.firstNewCodeButton.grid(row=1, column=1, sticky=tk.W+tk.E, padx=(5,15), pady=(5,5))

        self.firstSeventyoneVar = tk.IntVar()
        self.firstSeventyoneCheckbutton = ttk.Checkbutton(self.codeSelFrame, text="771", onvalue=1, offvalue=0, variable=self.firstSeventyoneVar, command=lambda: self.insert_class(0, 771, self.firstSeventyoneVar.get()))
        self.firstSeventyoneCheckbutton.grid(row=2, column=0, sticky=tk.W+tk.E, padx=(30,15), pady=(5,5))

        self.firstSeventythreeVar = tk.IntVar()
        self.firstSeventythreeCheckbutton = ttk.Checkbutton(self.codeSelFrame, text="773", onvalue=1, offvalue=0, variable=self.firstSeventythreeVar, command=lambda: self.insert_class(0, 773, self.firstSeventythreeVar.get()))
        self.firstSeventythreeCheckbutton.grid(row=3, column=0, sticky=tk.W+tk.E, padx=(30,15), pady=(5,5))

        self.firstSeventyVar = tk.IntVar()
        self.firstSeventyCheckbutton = ttk.Checkbutton(self.codeSelFrame, text="770", onvalue=1, offvalue=0, variable=self.firstSeventyVar, command=lambda: self.insert_class(0, 770, self.firstSeventyVar.get()))
        self.firstSeventyCheckbutton.grid(row=4, column=0, sticky=tk.W+tk.E, padx=(30,15), pady=(5,30))



        #second column
        self.secondClassLabel = ttk.Label(self.codeSelFrame, text="Second Class Codes:")
        self.secondClassLabel.grid(row=0, column=2, sticky=tk.W+tk.E, padx=15, pady=(5,5))

        self.secondNewVar = tk.IntVar()
        self.secondNewCodeEntry = ttk.Entry(self.codeSelFrame, textvariable=self.secondNewVar)
        self.secondNewCodeEntry.grid(row=1, column=2, sticky=tk.W+tk.E, padx=(15,5), pady=(5,5))
        self.secondNewCodeButton = ttk.Button(self.codeSelFrame, text="Add", command=lambda: self.insert_class(1, self.secondNewVar.get(), 1))
        self.secondNewCodeButton.grid(row=1, column=3, sticky=tk.W+tk.E, padx=(5,15), pady=(5,5))

        self.secondSeventyoneVar = tk.IntVar()
        self.secondSeventyoneCheckbutton = ttk.Checkbutton(self.codeSelFrame, text="771", onvalue=1, offvalue=0, variable=self.secondSeventyoneVar, command=lambda: self.insert_class(1, 771, self.secondSeventyoneVar.get()))
        self.secondSeventyoneCheckbutton.grid(row=2, column=2, sticky=tk.W+tk.E, padx=15, pady=(5,5))

        self.secondSeventythreeVar = tk.IntVar()
        self.secondSeventythreeCheckbutton = ttk.Checkbutton(self.codeSelFrame, text="773", onvalue=1, offvalue=0, variable=self.secondSeventythreeVar, command=lambda: self.insert_class(1, 773, self.secondSeventythreeVar.get()))
        self.secondSeventythreeCheckbutton.grid(row=3, column=2, sticky=tk.W+tk.E, padx=15, pady=(5,5))

        self.secondSeventyVar = tk.IntVar()
        self.secondSeventyCheckbutton = ttk.Checkbutton(self.codeSelFrame, text="770", onvalue=1, offvalue=0, variable=self.secondSeventyVar, command=lambda: self.insert_class(1, 770, self.secondSeventyVar.get()))
        self.secondSeventyCheckbutton.grid(row=4, column=2, sticky=tk.W+tk.E, padx=15, pady=(5,30))


        self.dialCheckbutton = ttk.Checkbutton(self.guiOptionsFrame, text="Load Data", variable=self.load_data, onvalue=1, offvalue=0)
        self.dialCheckbutton.grid(row=0, column=0, sticky=tk.W, padx=30, pady=(20,5))

        self.logVar = tk.IntVar()
        self.logCheckbutton = ttk.Checkbutton(self.guiOptionsFrame, text="Log data", variable=self.logVar, onvalue=1, offvalue=0)
        self.logCheckbutton.grid(row=1, column=0, sticky=tk.W, padx=30, pady=(5,5))



        self.launchButton = ttk.Button(self.guiOptionsFrame, text="Launch Training", command=self.quit)
        self.launchButton.grid(row=5, column=0, sticky=tk.N+tk.S+tk.W+tk.E, padx=30, pady=20, ipadx=10, ipady=10)

        #end of init
