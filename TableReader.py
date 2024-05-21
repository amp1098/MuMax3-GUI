import tkinter as tk
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import *
from tkinter import ttk
import ctypes
from tkinter.filedialog import askopenfile
from tkinter import filedialog
import numpy as np
import scipy.fftpack as sf
import pandas as pd
from scipy.signal import find_peaks, butter, lfilter, iirnotch
import os

# This code takes a table outputted by MuMax3 and analyzes the signals from it. It was made for structures that contain
# two regions (an antenna and a detector), but really it can read time-dependent data, FFT it and find the peaks.

# --- creating root window ---
ctypes.windll.shcore.SetProcessDpiAwareness(1)  # this makes all the windows match the device resolution

root = tk.Tk()
root.title("Spin Wave Signal Analyzer")
root.geometry('1440x800')

frame = tk.Frame(root)
frame.grid(row=0, column=0, sticky=NW)
root.grid_rowconfigure(0, weight=5)
root.grid_columnconfigure(0, weight=2)

plot_frame = tk.Frame(root)
plot_frame.grid(row=0, column=1, rowspan=1, columnspan=2, stick=NW)
root.grid_columnconfigure(1, weight=1)

toolbarFrame = tk.Frame(root)
toolbarFrame.grid(row=2, column=0, columnspan=2, sticky=W)

input_col_1_var = tk.IntVar(name="InCol1", value=0)
input_col_2_var = tk.IntVar(name="InCol2", value=1)
fft_scale_var = tk.IntVar(name="ScaleFFT", value=4)
peak_num_var = tk.IntVar(name='PeakNum', value=3)

vars_list = [input_col_1_var, input_col_2_var]

# --- Data Reading from Table ---

col_names = []

data = None


def reader(fldr):
    global data
    table = pd.read_csv(fldr, delimiter='\t')

    for col in table.columns:
        col_names.append(col)

    data = table.to_numpy()

    return data


def open_file():
    file = askopenfile(mode='r', filetypes=[('Text Files', '*.txt')])
    if file is not None:
        fldr = file.name
        reader(fldr)
        update()

        return fldr


def open_folder():
    file = filedialog.askdirectory()
    if file is not None:
        return file


t_path = open_folder() + "/table.txt"
output_path = open_folder()
reader(t_path)


# # --- Subtracting Two Tables To Find Characteristic Signal ---
#
# fldr_2 = "E:/Users/ChaseH/Documents/MainFiles/AcademicStuff/Physics Research/FFT Results/CRE Analysis/Characteristic Data"
#
# table_1 = pd.read_csv(fldr_2 + '/table_1.txt', delimiter='\t')
# table_2 = pd.read_csv(fldr_2 + '/table_2.txt', delimiter='\t')
#
# table_sub = table_1.subtract(table_2)
#
# table_sub['# t (s)'] = table_2['# t (s)']
#
# col_names = []
# for col in table_sub.columns:
#     col_names.append(col)
#
# table_sub = table_sub.dropna()
#
# data = table_sub.to_numpy()
#

# --- Labels and Entry Fields ---

labels = ["Input Column 1", "Input Column 2", "FFT X-Axis Scale", "Number of Peaks Plotted"]

for thing in labels:  # iterating through labels and vars_list to dynamically create Label and Entry objects
    tk.Label(frame, text=thing).grid(sticky="W", row=labels.index(thing), column=0)

combobox_1_var = StringVar(value=col_names[col_names.index('mx ()')])
combobox_2_var = StringVar(value=col_names[col_names.index('mz ()')])


input_1_column_box = ttk.Combobox(master=frame, textvariable=combobox_1_var, values=col_names, width=14)
input_1_column_box.grid(row=0, column=1)

input_2_column_box = ttk.Combobox(master=frame, textvariable=combobox_2_var, values=col_names, width=14)
input_2_column_box.grid(row=1, column=1)

scale_entry = tk.Entry(frame, textvariable=fft_scale_var, width=10)
scale_entry.grid(row=labels.index("FFT X-Axis Scale"), column=1)

peak_num_entry = tk.Entry(frame, textvariable=peak_num_var, width=4)
peak_num_entry.grid(row=labels.index("Number of Peaks Plotted"), column=1)


# --- figures/frames to be placed in the root window ---
fig = Figure(figsize=(8, 7), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=plot_frame)  # A tk.DrawingArea.

# creating an empty plot pack all the widgets in on startup; without this, everything only loads when enter is pressed
# fig.add_subplot().plot([], [])
# fig.add_subplot().set_title("Press Update When Ready")
canvas.draw()
canvas.get_tk_widget().grid(row=0, column=0)

toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)  # helpful toolbar to let people zoom in and stuff
toolbar.grid(row=1, column=2, ipadx=10)

# --- Variables to help with peak finding ---

peaks_1, peaks_2 = ([], [])  # currently unused

# --- Functions to help FFT stuff ---


def fft(signal, col_num):

    # remember, for [:, 0] the zero refers to the column number for the time data
    signal_size = signal[:, 0].size  # number of data points
    time_first = signal[0, 0]  # first time value
    time_final = signal[signal_size - 1, 0]  # last time value
    time_array = signal[:, 0]  # array for time data
    input_array = signal[:, col_num]  # array for mz data
    timestep = 1

    x_int = np.linspace(time_first, time_final, signal_size)  # creates evenly space points
    y_int = np.interp(x_int, time_array, input_array)  # interpolates the data

    sample = signal_size / (time_final - time_first)  # Sample rate (measurements/second)
    FFT = sf.fft(y_int - y_int.mean())  # calculates FFT, subtract mean to get rid of spike at 0Hz
    f_val = sf.fftfreq(len(y_int)) * sample  # converts time into frequency

    return [time_array, input_array, f_val, FFT, time_first, signal_size]


# class FftSimple(self, *args, **kwargs):
#     def __init__(self):
#         self.name = name
#
#     def sig_size(self):
#         signal_size = len(self)
#         return signal_size
#
#     def times(self):
#         time_first = 0
#         time_final = 5e-9
#         time_array = np.linspace

def fft_simple(signal, time_max):
    signal_size = len(signal)  # number of data points
    time_first = 0  # first time value
    time_final = time_max  # last time value
    time_array = np.linspace(0, time_max, signal_size)  # array for time data
    input_array = signal  # array for mz data

    x_int = np.linspace(time_first, time_final, signal_size)  # creates evenly space points
    y_int = np.interp(x_int, time_array, input_array)  # interpolates the data

    sample = signal_size / (time_final - time_first)  # Sample rate (measurements/second)
    FFT = sf.fft(y_int - y_int.mean())  # calculates FFT, subtract mean to get rid of spike at 0Hz
    f_val = sf.fftfreq(len(y_int)) * sample  # converts time into frequency

    return [time_array, input_array, f_val, FFT, time_first, signal_size]


def add_tables(fldr, col_num):  # this will FFT all the tables in this folder and add the frequency data together

    filenames = []
    for item in os.listdir(fldr):
        if item.endswith('.txt'):
            filenames.append(item)

    tables = {}
    size = 0
    for file in filenames:
        size += 1
        tables[file] = pd.read_csv(fldr + "/" + file, delimiter='\t').to_numpy()

    fft_data = {}
    for key in tables:
        fft_data[key] = fft(tables[key], col_num)[3]

    int_fft = sum(fft_data.values())

    return int_fft


def butter_highpass(cutoff, f_sample, order=5):
    nyq = 0.5 * f_sample
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def butter_highpass_filter(data_input, cutoff, f_sample, order=5):
    b, a = butter_highpass(cutoff, f_sample, order=order)
    # this applies the filter coefficients to our data ? apparently? idk
    y = lfilter(b, a, data_input)
    return y


def butter_bandpass(f_min, f_max, f_sample, order=5):
    nyq = 0.5 * f_sample
    normal_cutoff = [f_min / nyq, f_max / nyq]
    b, a = butter(order, normal_cutoff, btype="bandpass", analog=False)
    return b, a


def butter_bandpass_filter(data_input, f_min, f_max, f_sample, order=5):
    b, a = butter_bandpass(f_min, f_max, f_sample, order=order)
    # this applies the filter coefficients to our data ? apparently? idk
    y = lfilter(b, a, data_input)
    return y


def notch(f_0, f_sample, quality):
    b, a = iirnotch(f_0, quality, f_sample)
    return b, a


def notch_filter(data_input, f_0, f_sample, quality):
    b, a = notch(f_0, f_sample, quality)
    y = lfilter(b, a, data_input)
    return y


# --- updates and plotting ---

# root-mean-square function


def rms(x):
    # we need to square each item in the input data and them all together
    sum = 0
    for o in x:
        sum += o ** 2
    # then we need to divide it by the total number of items in x
    ms = sum / len(x)
    # and finally we square root the mean-sum
    return np.sqrt(ms)


def update():

    fig.clear()

    input_col_1_var.set(col_names.index(combobox_1_var.get()))
    input_col_2_var.set(col_names.index(combobox_2_var.get()))

    # t_col = 0
    input_col_emitter = input_col_1_var.get()
    input_col_detector = input_col_2_var.get()

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(223)
    ax3 = fig.add_subplot(222)
    ax4 = fig.add_subplot(224)
    # ax5 = fig.add_subplot(325)

    # --- Data Handling ---

    # - Left Side -

    to_plot = fft(data, input_col_emitter)

    time_array = to_plot[0]
    input_array_emitter = to_plot[1]
    # f_val = to_plot[2]
    # FFT = to_plot[3]
    time_final = to_plot[4]
    data_size = to_plot[5]

    # trying to apply filter

    # f_min_cut = 4e9
    # f_max_cut = 5e9
    #
    # y = butter_bandpass_filter(input_array_emitter, f_min_cut, f_max_cut, 1e11, order=6)

    f_cut = 4.8e9
    f_samp = 1e11
    f_min = 7e9
    f_max = 8.5e9

    y_notch = notch_filter(input_array_emitter, f_cut, f_samp, 1)
    y = butter_bandpass_filter(y_notch, f_min, f_max, f_samp, order=6)

    f_val = fft_simple(y, 5e-9)[2]
    FFT = fft_simple(y, 5e-9)[3]

    freq_value = f_val[:data_size // fft_scale_var.get()] / 1e9
    freq_count = np.abs(FFT[:data_size // fft_scale_var.get()])

    df_1 = pd.DataFrame()
    df_1.insert(0, 't (s)', time_array)
    df_1.insert(1, 'Mz', input_array_emitter)
    df_1.to_csv(output_path + "/timedomain_" + str(col_names[col_names.index(combobox_1_var.get())]) + ".csv")

    df_2 = pd.DataFrame()
    df_2.insert(0, 'f (GHz)', freq_value)
    df_2.insert(1, 'count', freq_count)
    df_2.to_csv(output_path + "/freqdomain_" + str(col_names[col_names.index(combobox_1_var.get())]) + ".csv")

    # IMPORTANT : The resolution of the FFT data depends on the total number of data points. To increase the frequency
    # resolution, you must increase the time resolution.

    # finding all peaks

    global peaks_1
    peaks_1, _ = find_peaks(freq_count, height=0)

    # === FIND N-LARGEST PEAKS ===

    peaks_1_dict = {}

    for key in peaks_1:
        peaks_1_dict[key] = np.abs(FFT[key])

    peaks_1_sorted = sorted(peaks_1_dict.items(), key=lambda pair: pair[1], reverse=True)[:peak_num_var.get()]

    # === FIND N-LARGEST PEAKS ===

    # ax1.plot(time_array / 1e-9, input_array_emitter, linewidth=0.75)  # note: ns
    ax1.plot(time_array / 1e-9, y, linewidth=0.75)  # filtered with notch and bandpass filter
    ax1.plot(time_array / 1e-9, np.repeat(rms(y), len(time_array)),
             linestyle='dashed', color='black', linewidth=1)  # plotting RMS
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Amplitude (M^2)')
    ax1.grid(True)
    ax1.minorticks_on()
    ax1.set_title(combobox_1_var.get() + 'filtered')

    ax2.plot(freq_value, freq_count)  # note 1: GHz. note 2: only positive half

    for item in peaks_1_sorted:
        key = int(item[0])
        ax2.scatter(f_val[key] / 1e9, np.abs(FFT[key]), edgecolors='black', facecolors='none')  # peaks
        ax2.text(f_val[key] / 1e9, np.abs(FFT[key]), str(round(f_val[key] / 1e9, 4)), horizontalalignment='center',
                 verticalalignment='bottom')

    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Magnitude')
    ax2.grid(True)
    ax2.minorticks_on()

    # - Right Side -

    to_plot = fft(data, input_col_detector)

    time_array = to_plot[0]
    input_array_detector = to_plot[1]
    f_val = to_plot[2]
    FFT = to_plot[3]
    time_final = to_plot[4]
    data_size = to_plot[5]

    freq_value = f_val[:data_size // fft_scale_var.get()] / 1e9
    freq_count = np.abs(FFT[:data_size // fft_scale_var.get()])

    df_3 = pd.DataFrame()
    df_3.insert(0, 't (s)', time_array)
    df_3.insert(1, 'Mz', input_array_detector)
    df_3.to_csv(output_path + "/timedomain_" + str(col_names[col_names.index(combobox_2_var.get())]) + ".csv")

    df_4 = pd.DataFrame()
    df_4.insert(0, 'f (GHz)', freq_value)
    df_4.insert(1, 'count', freq_count)
    df_4.to_csv(output_path + "/freqdomain_" + str(col_names[col_names.index(combobox_2_var.get())]) + ".csv")

    # # finding all peaks
    global peaks_2
    peaks_2, _ = find_peaks(freq_count, height=0)

    # === FIND N-LARGEST PEAKS ===

    peaks_2_dict = {}

    for key in peaks_2:
        peaks_2_dict[key] = np.abs(FFT[key])

    peaks_2_sorted = sorted(peaks_2_dict.items(), key=lambda pair: pair[1], reverse=True)[:peak_num_var.get()]

    # === FIND N-LARGEST PEAKS ===

    ax3.plot(time_array / 1e-9, input_array_detector, linewidth=0.75)  # note: ns
    ax3.plot(time_array / 1e-9, np.repeat(rms(input_array_detector), len(time_array)),
             linestyle='dashed', color='black', linewidth=1)  # plotting RMS
    ax3.set_xlabel('Time (ns)')
    ax3.set_ylabel('Amplitude (M^2)')
    ax3.grid(True)
    ax3.minorticks_on()
    ax3.set_title(combobox_2_var.get())

    ax4.plot(freq_value, freq_count)  # note 1: GHz. note 2: only positive half

    for item in peaks_2_sorted:
        key = int(item[0])
        ax4.scatter(f_val[key] / 1e9, np.abs(FFT[key]), edgecolors='black', facecolors='none')  # peaks
        ax4.text(f_val[key] / 1e9, np.abs(FFT[key]), str(round(f_val[key] / 1e9, 4)), horizontalalignment='center',
                 verticalalignment='bottom')

    ax4.set_xlabel('Frequency (GHz)')
    ax4.set_ylabel('Magnitude')
    ax4.grid(True)
    ax4.minorticks_on()

    # # - Integrating all frequency data -
    #
    # ax5.plot(f_val[:data_size // fft_scale_var.get()] / 1e9, np.abs(add_tables(folder, input_col_detector)[:data_size // fft_scale_var.get()]))
    # ax5.set_xlabel('Frequency (GHz)')
    # ax5.set_ylabel('Magnitude')
    # ax5.set_title("Sum of FFT Data")
    # ax5.grid(True)
    # ax5.minorticks_on()

    fig.tight_layout()

    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0)


# --- buttons ---

load_graph_button = Button(master=frame, text="Load Graph", command=update)
load_graph_button.grid(row=4, column=0)

# find_peaks_button = Button(master=frame, text="Find Signal Peaks", command=peak_plot)
# find_peaks_button.grid(row=4, column=0)

file_button = Button(master=frame, text="Browse Table Files", command=open_file)
file_button.grid(row=5, column=0)

# folder_button = Button(master=frame, text="Browse Table Folders", command=open_folder)
# folder_button.grid(row=6, column=0)

root.mainloop()
