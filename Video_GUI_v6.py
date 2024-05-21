import tkinter as tk

import matplotlib
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
import ctypes
from tkinter.filedialog import askopenfile
from tkinter import filedialog
import numpy as np
import numpy.ma as ma
import time
import sys
import os
import pandas as pd
import pickle
import threading
from matplotlib.animation import FFMpegWriter
import matplotlib.animation as ani
import matplotlib.colors as colors
import scipy.fftpack as sf
from PIL import Image


# --- creating root window ---
ctypes.windll.shcore.SetProcessDpiAwareness(1)  # this makes all the windows match the device resolution

# root window, contains frames

root = tk.Tk()
root.title("MuMax3 Video Making Suite")
root.geometry('1000x650')

# frame object, this is where the menus and pictures will go

frame = tk.Frame(root)
frame.grid(row=0, column=0, sticky=NW)

root.grid_rowconfigure(0, weight=5)
root.grid_columnconfigure(0, weight=2)

toolbarFrame = tk.Frame(root)
toolbarFrame.grid(row=3, column=1, columnspan=2, sticky=W)

# menu stuff
menubar = tk.Menu(root)
filemenu = tk.Menu(menubar, tearoff=0)
Imagemenu = tk.Menu(menubar, tearoff=0)
Videomenu = tk.Menu(menubar, tearoff=0)
helpmenu = tk.Menu(menubar, tearoff=0)

timer = tk.Label(frame, width=30, height=20, text="Total time: ")
timer.grid(row=20, column=1, columnspan=1, sticky=W)

progressbar = ttk.Progressbar(frame)
progressbar.grid(row=20, column=0, columnspan=2, sticky=W)

# for counting time it takes to cache things
start_time = None
elapsed_time = None

# main starting folder variable
folder = None

# filenames list
filenames = []

# column names list
col_names = []

# filename dictionary to store names of ovf files
name_dictionary = {}

# Variable to define whether 3d renders are enabled
ThreeD = False

# cmap variables
cmap_surf = None
cmap_top = None

# magnetization component variables
mx_var = tk.IntVar()
my_var = tk.IntVar()
mz_var = tk.IntVar(value=1)

# log limit values
lim_min = tk.StringVar(value="1e-6")
lim_max = tk.StringVar(value="0.25")

# video name variable
vid_name = tk.StringVar(value="your_video.mp4")

# current ovf file for renderings of single images, allows user to click between frames
current_ovf = tk.IntVar(value=0)

# DPI values
DPI_val = tk.IntVar(value=150)

# FPS values
FPS_val = tk.IntVar(value=24)

# vector field step size
step_val = tk.IntVar(value=100)

# coretracking toggle variable and list of points for trajectory plotting
coretrack_var = tk.BooleanVar(value=False)  # true = core tracking on, false = core tracking off
coretrack_state_var = tk.BooleanVar(value=True)  # true = 'up', false = 'down'
coretracker_xlist = []
coretracker_ylist = []


def convert_time(time_in_seconds):
    # returns time in list of [s, m, h] to be used for sec:min:hour format
    seconds_total = int(time_in_seconds % 60)
    minutes_total = int(((seconds_total - time_in_seconds) / -60) % 60)
    hours_total = int(((minutes_total - ((seconds_total - time_in_seconds) / -60)) / -60) % 60)

    return [seconds_total, minutes_total, hours_total]


def timing_function(items, current):
    # tells how long an iterative task will take
    # items arg is a set of objects to be iterated through
    # item is an instance of items defined by index current
    global start_time, elapsed_time
    item = items[current]

    # file packing time estimation stuff :-)

    if items[0] == item:
        start_time = time.perf_counter()

    elif items[1] == item:

        # find how long one item takes to iterate through, then multiply that by the total
        # gives a good estimate of total time if the iteration is uniform
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time


def exp_find(n):
    return np.floor(np.log10(n))


# --- commands for buttons ---

def open_file():
    file = askopenfile(mode='r', filetypes=[('OVF Files', '*.ovf')])
    if file is not None:
        fldr = file.name

        return fldr


def open_folder():
    global folder

    user_input_folder = filedialog.askdirectory()
    if user_input_folder is None:

        return
    folder = user_input_folder

    return folder


def create_ovf_cache():
    # generates a file containing specified OVF files to be later accessed; speeds up repeated video/image making
    global folder, elapsed_time, name_dictionary, filenames
    print("No cached dataframe dictionary detected, generating a new one...")

    if folder is None:
        open_folder()
    else:
        for item in os.listdir(folder):
            # if folder starts with any of these, ignore it
            if item.startswith(('FullInitialState', 'Geometry', 'InitialMagnetization', 'Regions')):
                pass
            # if folder ends with .ovf, add it to filenames list
            elif item.endswith('.ovf'):
                filenames.append(item)

    # We can use a dictionary to loop through the names and create multiple dataframes based on each one
    # this reads each ovf file and saves it to the dictionary name_dict for later reference
    # the keys for each item in the dictionary start at m000000

    for name in filenames[1:]:
        df = pd.read_csv(folder + "/" + name, sep=" ", header=0, skiprows=28, names=['Mx', 'My', 'Mz', 'null'],
                         skipfooter=2, engine='python', usecols=['Mx', 'My', 'Mz'])
        df += 0  # removing signed zeros, it messes up the colormap
        name_dictionary[name] = df
        # find how long each iteration takes to finish in seconds
        timing_function(filenames[1:], filenames[1:].index(name))

        if elapsed_time is not None:

            # show how much longer you have to wait
            total_time = elapsed_time * len(filenames)
            current_time = elapsed_time * filenames.index(name)
            time_left = total_time - current_time
            conv_time_cache = convert_time(time_left)
            timer.configure(text=("Total time: " + str(conv_time_cache[0]) + "s " + str(conv_time_cache[1]) + "min " + str(
                conv_time_cache[2]) + "hr "))
        # increment loading bar by one unit, normalized by number of files
        step_size = 1 / len(filenames) * 100
        progressbar.step(step_size)

    # serializing
    print("Serializing cache...")
    full_path_dict = os.path.join(folder, "dict.pkl")
    with open(full_path_dict, "wb") as handle:  # the "wb" means it is writing this file as a binary
        pickle.dump(name_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Serialization complete!")


def start_cache():
    # threading necessary so GUI doesn't lock up
    thread = threading.Thread(target=create_ovf_cache, daemon=True)
    thread.start()


def detect_cache():
    if folder is None:
        open_folder()

        # check if dictionary pickle file is present
        for item in os.listdir(folder):
            if item == "dict.pkl":
                return True
        return False
    else:
        # check if dictionary pickle file is present

        for item in os.listdir(folder):
            if item == "dict.pkl":
                return True
        return False


def load_cache():
    global name_dictionary
    if len(filenames) != 0:
        return
    if detect_cache():
        print("Cached dataframe dictionary detected!")
        full_path = os.path.join(folder, "dict.pkl")
        with open(full_path, "rb") as handle:  # the "r" means it is reading this file as a binary
            name_dictionary = pickle.load(handle)
        for obj in name_dictionary:
            filenames.append(obj)

    else:
        print("start cache started")
        start_cache()


def retrieve_header_info():
    load_cache()

    # reading header info inside OVF files and assigning to list
    if not detect_cache():
        headers = pd.read_csv(folder + "/" + "m000000.ovf", sep=" ",
                              header=0, nrows=28,
                              names=['A', 'Label', 'Value'], engine='python', on_bad_lines='skip')
        headers.drop('A', axis=1, inplace=True)  # this drops the pound symbols column
        name_dictionary['headers'] = headers  # saves header info into dict.pikl file for future

    # grabs header info after cache is loaded
    x_size = int(name_dictionary['headers'].loc[17]['Value'])
    y_size = int(name_dictionary['headers'].loc[18]['Value'])
    z_size = int(name_dictionary['headers'].loc[19]['Value'])

    cellx = float(name_dictionary['headers'].loc[8]['Value'].split("e")[0]) * 1e-9
    celly = float(name_dictionary['headers'].loc[9]['Value'].split("e")[0]) * 1e-9
    return [x_size, y_size, z_size, cellx, celly]


def set_colormaps():
    global cmap_surf, cmap_top
    cmap_top = plt.get_cmap('hsv')
    cmap_top.set_bad(color='k', alpha=0.0)  # transparent if value is 0

    cmap_surf = plt.get_cmap('twilight')
    cmap_surf.set_bad(color='k', alpha=0.0)  # transparent if value is 0


class FasterFFMpegWriter(FFMpegWriter):
    # Faster video encoding, taken from
    # https://stackoverflow.com/questions/30965355/speedup-matplotlib-animation-to-video-file
    # this speeds up video encoding 3x, given my current hardware.
    '''FFMpeg-pipe writer bypassing figure.savefig.'''
    def __init__(self, **kwargs):
        '''Initialize the Writer object and sets the default frame_format.'''
        super().__init__(**kwargs)
        self.frame_format = 'argb'

    def grab_frame(self, **savefig_kwargs):
        '''Grab the image information from the figure and save as a movie frame.

        Doesn't use savefig to be faster: savefig_kwargs will be ignored.
        '''
        try:
            # re-adjust the figure size and dpi in case it has been changed by the
            # user.  We must ensure that every frame is the same size or
            # the movie will not save correctly.
            self.fig.set_size_inches(self._w, self._h)
            self.fig.set_dpi(self.dpi)
            # Draw and save the frame as an argb string to the pipe sink
            self.fig.canvas.draw()
            self._proc.stdin.write(self.fig.canvas.tostring_argb())
        except (RuntimeError, IOError) as e:
            out, err = self._proc.communicate()
            raise IOError('Error saving animation to file (cause: {0}) '
                      'Stdout: {1} StdError: {2}. It may help to re-run '
                      'with --verbose-debug.'.format(e, out, err))


def signif(x, p):
    # sig figs funciton
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


def Norm(x, y):
    return np.sqrt(x ** 2 + y ** 2)


def cent_diff(u, v):
    Dx = 512
    Dy = 512
    # plots curl numerical using central difference method, doesn't work on stuff at the edge yet
    w = np.empty((len(X), len(Y)))
    for i in range(len(u) - 1):
        for j in range(len(v[0]) - 1):
            w_ij = (v[i+1][j]-v[i-1][j])/(2*Dx) - (u[i][j+1]-u[i][j-1])/(2*Dy)
            np.append(w, w_ij)
    return w


def coretracker_state():
    if coretrack_state_var.get():
        return 'up'
    else:
        return 'down'


# --- Table Reading (For External Field and Time) ---
open_folder()
data = pd.DataFrame()

if folder is not None:
    table = pd.read_csv(folder + '/table.txt', delimiter='\t')

    for col in table.columns:
        col_names.append(col)


def col_list_generator():
    global data

    for thing in col_names:
        table_array = table[thing].to_numpy()
        data[thing] = table_array.tolist()


col_list_generator()

# === initializing figures for plotting ===

x_size, y_size, z_size, cellx, celly = retrieve_header_info()

if ThreeD:
    load_cache()
    fig = plt.figure(figsize=plt.figaspect(0.8,), dpi=DPI_val.get(), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')

    # turn off axis
    ax.set_axis_off()
else:
    if not detect_cache():
        aspect = 1
    else:
        aspect = 2*retrieve_header_info()[1]/retrieve_header_info()[0]
    fig = plt.figure(figsize=plt.figaspect(aspect,), dpi=DPI_val.get(), constrained_layout=True)
    ax = fig.add_subplot(211)
    ax_timeseries = fig.add_subplot(212)

    # turn off axis
    ax.set_axis_off()

    plot_frame = tk.Frame(root)
    plot_frame.grid(row=0, column=2, rowspan=1, columnspan=1, stick=NE)

    toolbarFrame = tk.Frame(root)
    toolbarFrame.grid(row=2, column=0, columnspan=2, sticky=W)

    root.grid_columnconfigure(1, weight=1)
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0)

    toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)  # helpful toolbar to let people zoom in and stuff
    toolbar.grid(row=1, column=2, ipadx=10)


def check_mag_comp():
    if mx_var.get() == 1:
        # if you want mx to be shown
        my_var.set(0)
        mz_var.set(0)
    elif my_var.get() == 1:
        # if you want mx to be shown
        mx_var.set(0)
        mz_var.set(0)
    elif mz_var.get() == 1:
        # if you want mx to be shown
        mx_var.set(0)
        my_var.set(0)


def toggle_coretracker():
    if coretrack_state_var.get():
        coretrack_state_var.set(False)
        coretrack_state_strvar.set("down")
    else:
        coretrack_state_var.set(True)
        coretrack_state_strvar.set("up")


coretrack_state_strvar = tk.StringVar(value=coretracker_state())


def old_coretracker(polarity):
    # perhaps I could tell this to ignore values to the left or right of a certain x-index?

    max_index_list = []
    max_point_list = []

    for key in filenames[1:]:
        # we construct our Mz array from the unpacked ovf file
        Mz = name_dictionary[key]['Mz'].to_numpy()
        # print("Mz: " + str(Mz))
        # we make it a 2D array with dimensions x_size, y_size
        Mz = Mz.reshape(y_size, x_size)
        # we find the max value in this array (or min if the core is down)
        # note that argmax returns a flat index (i.e. it just counts the elements until it find a maximum)
        # so you have to reshape it)

        # we could mask the values that are not in the area we are looking at
        # i just told the program to mask all array elements that are further than halfway across the entire structure

        mask_arr = np.empty(x_size)
        for j in np.arange(0, y_size - 1):
            masked_row = []
            for i in range(x_size):
                if i > x_size:
                    element = 1  # mask is on here
                else:
                    element = 0  # mask is off here
                masked_row.append(element)
            mask_arr = np.vstack([mask_arr, masked_row])

        Mz = ma.masked_array(Mz, mask_arr)

        if str(polarity).lower() == 'up':
            Mz_max_index = np.unravel_index(Mz.argmax(), np.shape(Mz))
        elif str(polarity).lower() == 'down':
            Mz_max_index = np.unravel_index(Mz.argmin(), np.shape(Mz))
        else:
            print("Argument for coretracker not understood. Use 'up' or 'down'.")
            break


        # gives a 2D point where the max is located in each dataframe object (i.e. in each OVF file)
        # keep in mind that the indices of spatial data relate to x and y coordinates, but the first index in an array
        # tells you the row array, which in a 2D array is more like a vertical position. So we need to swap the first
        # and second indices to make this correct. This is why we use (stuff[1], stuff[0]) and not (stuff[0], stuff[1]).
        Mz_max_index_loc = ([Mz_max_index][0][1], [Mz_max_index][0][0])
        # this returns the physical location of the core on the sample relative to the top left corner
        Mz_max_real_loc = (cellx * [Mz_max_index][0][1], celly * [Mz_max_index][0][0])
        # appends list with location
        max_index_list.append(Mz_max_index_loc)
        max_point_list.append(Mz_max_real_loc)

    return [max_index_list, max_point_list]


def coretracker(polarity, frame_num):
    # perhaps I could tell this to ignore values to the left or right of a certain x-index?

    # we construct our Mz array from the unpacked ovf file
    Mz = name_dictionary[frame_num]['Mz'].to_numpy(dtype=float)

    # print("Mz: " + str(Mz))
    # we make it a 2D array with dimensions x_size, y_size
    Mz = Mz.reshape(y_size, x_size)

    if str(polarity).lower() == 'up':
        Mz_max_index = np.unravel_index(Mz.argmax(), np.shape(Mz))
    elif str(polarity).lower() == 'down':
        Mz_max_index = np.unravel_index(Mz.argmin(), np.shape(Mz))
    else:
        print("Argument for coretracker not understood. Use 'up' or 'down'.")

    # gives a 2D point where the max is located in each dataframe object (i.e. in each OVF file)
    # keep in mind that the indices of spatial data relate to x and y coordinates, but the first index in an array
    # tells you the row array, which in a 2D array is more like a vertical position. So we need to swap the first
    # and second indices to make this correct. This is why we use (stuff[1], stuff[0]) and not (stuff[0], stuff[1]).
    Mz_max_index_loc = ([Mz_max_index][0][1], [Mz_max_index][0][0])
    # this returns the physical location of the core on the sample relative to the top left corner
    Mz_max_real_loc = (cellx * [Mz_max_index][0][1], celly * [Mz_max_index][0][0])
    # appends list with location
    return [[Mz_max_index][0][1], [Mz_max_index][0][0]]


def old_report_coretracker(polarity):

    points = coretracker(polarity)[1]

    x_vals = []
    y_vals = []

    x_vals_scaled = []
    y_vals_scaled = []

    for i in points:
        x_vals.append(i[0])
        y_vals.append(i[1])

        x_vals_scaled.append(i[0] / 10 ** exp_find(cellx))
        y_vals_scaled.append(i[1] / 10 ** exp_find(celly))

    x_vals_array = np.asarray(x_vals, dtype=np.float32)
    y_vals_array = np.asarray(y_vals, dtype=np.float32)

    df = pd.DataFrame()
    # df.insert(0, 't', pd.read_csv(folder + '/table.txt', delimiter='\t')[col_names[0]].to_numpy())
    df.insert(0, 'x', x_vals_array)
    df.insert(1, 'y', y_vals_array)
    # df.to_csv(folder + "/MzMax_positional.csv")

    return [points, x_vals_scaled, y_vals_scaled]


def coretracker_export():
    points = np.empty(len(filenames[1:]))

    for i in range(len(filenames[1:])):
        np.append(points, coretracker(coretracker_state(), i))

    x_vals = []
    y_vals = []

    x_vals_scaled = []
    y_vals_scaled = []

    for i in points:
        x_vals.append(i[0])
        y_vals.append(i[1])

        x_vals_scaled.append(i[0] / 10 ** exp_find(cellx))
        y_vals_scaled.append(i[1] / 10 ** exp_find(celly))

    x_vals_array = np.asarray(x_vals, dtype=np.float32)
    y_vals_array = np.asarray(y_vals, dtype=np.float32)

    df = pd.DataFrame()
    df.insert(0, 't', table[col_names[0]].to_numpy())
    df.insert(1, 'x', x_vals_array)
    df.insert(2, 'y', y_vals_array)
    df.to_csv(folder + "/MzMax_positional.csv")


def return_structure_image():
    # --- Figuring out the structure and drawing up a rough image of it ---
    # The structure doesn't change, so looking at the first frame should tell us what the structure looks like
    mag_x = name_dictionary[filenames[1:][0]]['Mx']
    mag_y = name_dictionary[filenames[1:][0]]['My']
    mag_z = name_dictionary[filenames[1:][0]]['Mz']

    # adding all the elements in each series together; the structure will be wherever some magnetization vector has
    # a non-zero magnitude

    mag_total = mag_x.add(mag_y, fill_value=0).add(mag_z, fill_value=0).to_list()

    for item in range(len(mag_total)):
        if mag_total[item] != 0:  # if the val is not 0 (i.e. structure present) set the val to an RGB black tuple
            mag_total[item] = (255, 255, 255)
        else:  # if val is 0 (i.e. no structure present) set the val to an RGB white tuple
            mag_total[item] = (0, 0, 0)

    img = Image.new('RGB', (x_size, y_size))  # Create a new image
    img.putdata(mag_total)

    return img


def old_plot_coretracker():
    polarity = 'down'
    retrieve_header_info()
    fig.clear()
    ax = fig.add_subplot(111)

    # turn off axis
    ax.set_axis_off()

    structure = return_structure_image()

    # returns whole parts of cells
    cellx_w = int(cellx / 10 ** exp_find(cellx))
    celly_w = int(celly / 10 ** exp_find(celly))

    x_vals_scaled = report_coretracker(polarity)[1]
    y_vals_scaled = report_coretracker(polarity)[2]
    p_scale = 2
    p_color = 'black'
    l_scale = 1
    l_color = 'blue'

    ax.scatter(x_vals_scaled, y_vals_scaled, s=p_scale, c=p_color)
    ax.plot(x_vals_scaled, y_vals_scaled, linewidth=l_scale, c=l_color)

    ax.imshow(structure.resize((cellx_w * x_size, celly_w * y_size)))

    if canvas: canvas.get_tk_widget().pack_forget()  # clear previous canvas

    canvas.draw()
    canvas.get_tk_widget().pack()


def fft(signal, col_num, current_ovf):
    if current_ovf != 0:
        # remember, for [:, 0] the zero refers to the column number for the time data
        time_array = signal[col_names[0]][0:current_ovf]  # array for time data

        signal_size = time_array.size  # number of data points
        time_first = time_array[0]  # first time value
        time_final = time_array[signal_size - 1]  # last time value
        input_array = signal[col_names[col_num]][0:current_ovf]  # array for mz data

        x_int = np.linspace(time_first, time_final, signal_size)  # creates evenly space points
        y_int = np.interp(x_int, time_array, input_array)  # interpolates the data

        sample = signal_size / (time_final - time_first)  # Sample rate (measurements/second)
        FFT = sf.fft(y_int - y_int.mean())  # calculates FFT, subtract mean to get rid of spike at 0Hz
        f_val = sf.fftfreq(len(y_int)) * sample  # converts time into frequency

        return [time_array, input_array, f_val, FFT, time_first, signal_size]
    else:
        # remember, for [:, 0] the zero refers to the column number for the time data
        time_array = signal[col_names[0]]  # array for time data

        signal_size = time_array.size  # number of data points
        time_first = time_array[0]  # first time value
        time_final = time_array[signal_size - 1]  # last time value
        input_array = signal[col_names[col_num]]  # array for mz data

        x_int = np.linspace(time_first, time_final, signal_size)  # creates evenly space points
        y_int = np.interp(x_int, time_array, input_array)  # interpolates the data

        sample = signal_size / (time_final - time_first)  # Sample rate (measurements/second)
        FFT = sf.fft(y_int - y_int.mean())  # calculates FFT, subtract mean to get rid of spike at 0Hz
        f_val = sf.fftfreq(len(y_int)) * sample  # converts time into frequency

        return [time_array, input_array, f_val, FFT, time_first, signal_size]


def update(frames):
    global cmap_surf, cmap_top, fig, ax, filenames, x_size, y_size, z_size, cellx, celly
    temp_filenames = filenames[1:]

    if ThreeD:
        ax.cla()

        # turn off axis
        ax.set_axis_off()

        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

        # Get rid of the spines
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        # Get rid of the ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # making panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # slower but idk how else to do this
        ax.set(
            zlim=(-3, 1.5),
            xlim=(0, x_size),
            ylim=(0, y_size)
        )

        # scale sometimes needs to be divided by 10, seems to show up if cell size is below 2 digits long

        ax.set_xlabel('\n' + str(signif(cellx * x_size / 10, 5)) + 'um', labelpad=-15)
        ax.set_ylabel('\n' + str(signif(celly * y_size / 10, 5)) + 'um', labelpad=-15)
        ax.set_zlabel('Mz', labelpad=-13)

        key = filenames[frames]
        Mx = name_dictionary[key]['Mx'].to_numpy(dtype=float)
        My = name_dictionary[key]['My'].to_numpy(dtype=float)
        Mz = name_dictionary[key]['Mz'].to_numpy(dtype=float)

        if x_size > y_size:
            ax.set_box_aspect((x_size / y_size, 1, 1), zoom=2)
        elif y_size > x_size:
            ax.set_box_aspect((1, x_size / y_size, 1), zoom=2)
        else:
            ax.set_box_aspect((1, 1, 1), zoom=2)

        Mag_x = np.ma.masked_equal(Mx.reshape(y_size, x_size), 0)
        Mag_y = np.ma.masked_equal(My.reshape(y_size, x_size), 0)
        Mag_z = np.ma.masked_equal(Mz.reshape(y_size, x_size), 0)

        grid_x, grid_y = np.meshgrid(
            np.arange(stop=x_size),
            np.arange(stop=y_size),
        )

        M = np.arctan2(My, Mx).reshape(y_size, x_size)
        M_masked = np.ma.masked_equal(M, 0)

        set_colormaps()

        # plots surface representing Mz data
        ax.plot_surface(grid_x, grid_y, Mag_z, rstride=1, cstride=1, linewidth=0, antialiased=True,
                        cmap=cmap_surf)

        # plots contour mapping of Mx and My data to show in-plane angles
        ax.contourf(np.arange(stop=x_size), np.arange(stop=y_size), M_masked, 200, zdir='z', offset=-2.95,
                    cmap=cmap_top)

        # update progress bar
        timing_function(filenames, frames)
        if elapsed_time is not None:
            # show how much longer you have to wait
            total_time = elapsed_time * len(filenames)
            current_time = elapsed_time * frames
            time_left = total_time - current_time
            conv_time_cache = convert_time(time_left)
            timer.configure(
                text=("Total time: " + str(conv_time_cache[0]) + "s " + str(conv_time_cache[1]) + "min " + str(
                    conv_time_cache[2]) + "hr "))
        # increment loading bar by one unit, normalized by number of files
        step_size = 1 / len(filenames) * 100
        progressbar.step(step_size)

    else:
        # 2D animation
        ax.cla()
        ax_timeseries.cla()

        # turn off axis
        ax.set_axis_off()

        # slower but idk how else to do this
        ax.set(
            xlim=(0, x_size),
            ylim=(0, y_size)
        )

        # scale sometimes needs to be divided by 10, seems to show up if cell size is below 2 digits long

        ax.set_xlabel('\n' + str(signif(cellx * x_size / 10, 5)) + 'um', labelpad=-15)
        ax.set_ylabel('\n' + str(signif(celly * y_size / 10, 5)) + 'um', labelpad=-15)

        # unpacking data
        key = temp_filenames[frames]
        Mx = name_dictionary[key]['Mx'].to_numpy(dtype=float)
        My = name_dictionary[key]['My'].to_numpy(dtype=float)
        Mz = name_dictionary[key]['Mz'].to_numpy(dtype=float)
        Mag_x = np.ma.masked_equal(Mx.reshape(y_size, x_size), 0)
        Mag_y = np.ma.masked_equal(My.reshape(y_size, x_size), 0)
        Mag_z = np.ma.masked_equal(Mz.reshape(y_size, x_size), 0)

        grid_x, grid_y = np.meshgrid(
            np.arange(stop=x_size),
            np.arange(stop=y_size),
        )

        set_colormaps()

        min_val = float(lim_min.get())
        max_val = float(lim_max.get())

        step = step_val.get()

        (U, V) = Mag_x, Mag_y

        to_plot = fft(table, 3, frames)

        time_array = to_plot[0]
        signal_array = to_plot[1]

        if mx_var.get() == 1:
            ax.pcolormesh(grid_x, grid_y, Mag_x ** 2, cmap=cmap_surf, norm=colors.LogNorm(vmin=min_val, vmax=max_val))
            ax.quiver(grid_x[::step, ::step], grid_y[::step, ::step], U[::step, ::step], V[::step, ::step],
                  linewidth=0.5, edgecolor='black', facecolor='white')
            if coretrack_var.get():
                core_pos = coretracker(coretracker_state(), key)
                coretracker_xlist.append(core_pos[0])
                coretracker_ylist.append(core_pos[1])
                core_x = core_pos[0]
                core_y = core_pos[1]
                ax.scatter(core_x, core_y, s=5, c='white', marker='o')
                ax.plot(coretracker_xlist[3:frames], coretracker_ylist[3:frames], c='k',
                        linewidth=0.5)
        elif my_var.get() == 1:
            ax.pcolormesh(grid_x, grid_y, Mag_y ** 2, cmap=cmap_surf, norm=colors.LogNorm(vmin=min_val, vmax=max_val))
            ax.quiver(grid_x[::step, ::step], grid_y[::step, ::step], U[::step, ::step], V[::step, ::step],
                  linewidth=0.5, edgecolor='black', facecolor='white')
            if coretrack_var.get():
                core_pos = coretracker(coretracker_state(), key)
                coretracker_xlist.append(core_pos[0])
                coretracker_ylist.append(core_pos[1])
                core_x = core_pos[0]
                core_y = core_pos[1]
                ax.scatter(core_x, core_y, s=5, c='white', marker='o')
                ax.plot(coretracker_xlist[3:frames], coretracker_ylist[3:frames], c='k',
                        linewidth=0.5)
        elif mz_var.get() == 1:
            ax.pcolormesh(grid_x, grid_y, Mag_z ** 2, cmap=cmap_surf, norm=colors.LogNorm(vmin=min_val, vmax=max_val))
            ax.quiver(grid_x[::step, ::step], grid_y[::step, ::step], U[::step, ::step], V[::step, ::step],
                  linewidth=0.5, edgecolor='black', facecolor='white')
            if coretrack_var.get():
                core_pos = coretracker(coretracker_state(), key)
                coretracker_xlist.append(core_pos[0])
                coretracker_ylist.append(core_pos[1])
                core_x = core_pos[0]
                core_y = core_pos[1]
                ax.scatter(core_x, core_y, s=5, c='white', marker='o')
                ax.plot(coretracker_xlist[3:frames], coretracker_ylist[3:frames], c='k',
                        linewidth=0.5)

            ax_timeseries.plot(time_array / 1e-9, signal_array,
                               linewidth=0.75)  # filtered with notch and bandpass filter
            ax_timeseries.plot(time_array / 1e-9, np.repeat(rms(signal_array), len(time_array)),
                               linestyle='dashed', color='black', linewidth=1)  # plotting RMS
            ax_timeseries.set_xlabel('Time (ns)')
            ax_timeseries.set_ylabel('Amplitude (M^2)')
            ax_timeseries.grid(True)
            ax_timeseries.minorticks_on()
            ax_timeseries.set_title("M_z^2")
        # update progress bar
        timing_function(filenames, frames)
        if elapsed_time is not None:

            # show how much longer you have to wait
            total_time = elapsed_time * len(temp_filenames)
            current_time = elapsed_time * frames
            time_left = total_time - current_time
            conv_time_cache = convert_time(time_left)
            timer.configure(text=("Total time: " + str(conv_time_cache[0]) + "s " + str(conv_time_cache[1]) + "min " + str(
                conv_time_cache[2]) + "hr "))
        # increment loading bar by one unit, normalized by number of files
        step_size = 1 / len(filenames) * 100
        progressbar.step(step_size)

        if canvas: canvas.get_tk_widget().pack_forget()  # clear previous canvas

        canvas.draw()
        canvas.get_tk_widget().pack()


def clear_frame():
    canvas.get_tk_widget().pack_forget()


def rms(x):
    # we need to square each item in the input data and them all together
    sum = 0
    for o in x:
        sum += o ** 2
    # then we need to divide it by the total number of items in x
    ms = sum / len(x)
    # and finally we square root the mean-sum
    return np.sqrt(ms)


def time_int():
    retrieve_header_info()
    fig.clear()
    ax = fig.add_subplot(111)

    # turn off axis
    ax.set_axis_off()

    # array made of zeroes to add to
    sum_total = np.zeros(y_size * x_size)
    grid_x, grid_y = np.meshgrid(
        np.arange(stop=x_size),
        np.arange(stop=y_size),
    )

    for item in filenames[1:]:
        Mx = name_dictionary[item]['Mx'].to_numpy(dtype=float)
        My = name_dictionary[item]['My'].to_numpy(dtype=float)
        Mz = name_dictionary[item]['Mz'].to_numpy(dtype=float)

        if mx_var.get() == 1:
            sum_total = np.add(sum_total, Mx)
        elif my_var.get() == 1:
            sum_total = np.add(sum_total, My)
        elif mz_var.get() == 1:
            sum_total = np.add(sum_total, Mz)

    sum_total = np.ma.masked_equal(sum_total.reshape(y_size, x_size), 0)
    # scale sometimes needs to be divided by 10, seems to show up if cell size is below 2 digits long

    ax.set_xlabel('\n' + str(signif(cellx * x_size, 5)) + 'um', labelpad=-15)
    ax.set_ylabel('\n' + str(signif(celly * y_size, 5)) + 'um', labelpad=-15)

    min_val = float(lim_min.get())
    max_val = float(lim_max.get())

    ax.pcolormesh(grid_x, grid_y, sum_total ** 2, cmap=cmap_surf, norm=colors.LogNorm(
        vmin=min_val, vmax=max_val)
                  )

    if canvas: canvas.get_tk_widget().pack_forget()  # clear previous canvas

    canvas.draw()
    canvas.get_tk_widget().pack()


def test_command():
    print("This works!")


def help_popout():
    top = Toplevel(root)
    top.geometry("500x250")
    top.title("Help Window")
    Label(top, text="This GUI allows for a user to analyze folders containing OVF files generated by MuMax3. The \n"
                    + "features currently included are: static and animated OVF visualization, vortex core tracking \n"
                    + "and time-integrated image generation.").pack()


def plot_current_ovf():
    temp_filenames = filenames[1:]

    # 2D animation
    ax.cla()
    ax_timeseries.cla()

    # turn off axis
    ax.set_axis_off()

    # slower but idk how else to do this
    ax.set(
        xlim=(0, x_size),
        ylim=(0, y_size)
    )

    # scale sometimes needs to be divided by 10, seems to show up if cell size is below 2 digits long

    ax.set_xlabel('\n' + str(signif(cellx * x_size / 10, 5)) + 'um', labelpad=-15)
    ax.set_ylabel('\n' + str(signif(celly * y_size / 10, 5)) + 'um', labelpad=-15)

    # unpacking data
    key = temp_filenames[current_ovf.get()]
    Mx = name_dictionary[key]['Mx'].to_numpy(dtype=float)
    My = name_dictionary[key]['My'].to_numpy(dtype=float)
    Mz = name_dictionary[key]['Mz'].to_numpy(dtype=float)
    Mag_x = np.ma.masked_equal(Mx.reshape(y_size, x_size), 0)
    Mag_y = np.ma.masked_equal(My.reshape(y_size, x_size), 0)
    Mag_z = np.ma.masked_equal(Mz.reshape(y_size, x_size), 0)

    grid_x, grid_y = np.meshgrid(
        np.arange(stop=x_size),
        np.arange(stop=y_size),
    )

    set_colormaps()

    min_val = float(lim_min.get())
    max_val = float(lim_max.get())

    (U, V) = Mag_x, Mag_y

    step = step_val.get()
    to_plot = fft(table, 3, current_ovf.get())

    time_array = to_plot[0]
    signal_array = to_plot[1]

    if mx_var.get() == 1:
        ax.pcolormesh(grid_x, grid_y, Mag_x ** 2, cmap=cmap_surf, norm=colors.LogNorm(vmin=min_val, vmax=max_val))
        ax.quiver(grid_x[::step, ::step], grid_y[::step, ::step], U[::step, ::step], V[::step, ::step],
                  linewidth=0.5, edgecolor='black', facecolor='white')
        if coretrack_var.get():
            core_pos = coretracker(coretracker_state(), key)
            coretracker_xlist.append(core_pos[0])
            coretracker_ylist.append(core_pos[1])
            core_x = core_pos[0]
            core_y = core_pos[1]
            ax.scatter(core_x, core_y, s=5, c='white', marker='o')
            ax.plot(coretracker_xlist[0:current_ovf.get()], coretracker_ylist[0:current_ovf.get()], c='k',
                    linewidth=0.5)
    elif my_var.get() == 1:
        ax.pcolormesh(grid_x, grid_y, Mag_y ** 2, cmap=cmap_surf, norm=colors.LogNorm(vmin=min_val, vmax=max_val))
        ax.quiver(grid_x[::step, ::step], grid_y[::step, ::step], U[::step, ::step], V[::step, ::step],
                  linewidth=0.5, edgecolor='black', facecolor='white')
        if coretrack_var.get():
            core_pos = coretracker(coretracker_state(), key)
            coretracker_xlist.append(core_pos[0])
            coretracker_ylist.append(core_pos[1])
            core_x = core_pos[0]
            core_y = core_pos[1]
            ax.scatter(core_x, core_y, s=5, c='white', marker='o')
            ax.plot(coretracker_xlist[0:current_ovf.get()], coretracker_ylist[0:current_ovf.get()], c='k',
                    linewidth=0.5)
    elif mz_var.get() == 1:
        ax.pcolormesh(grid_x, grid_y, Mag_z ** 2, cmap=cmap_surf, norm=colors.LogNorm(vmin=min_val, vmax=max_val))
        ax.quiver(grid_x[::step, ::step], grid_y[::step, ::step], U[::step, ::step], V[::step, ::step],
                  linewidth=0.5, edgecolor='black', facecolor='white')
        if coretrack_var.get():
            core_pos = coretracker(coretracker_state(), key)
            coretracker_xlist.append(core_pos[0])
            coretracker_ylist.append(core_pos[1])
            core_x = core_pos[0]
            core_y = core_pos[1]
            ax.scatter(core_x, core_y, s=5, c='white', marker='o')
            ax.plot(coretracker_xlist[0:current_ovf.get()], coretracker_ylist[0:current_ovf.get()], c='k',
                    linewidth=0.5)

        ax_timeseries.plot(time_array / 1e-9, signal_array, linewidth=0.75)  # filtered with notch and bandpass filter
        ax_timeseries.plot(time_array / 1e-9, np.repeat(rms(signal_array), len(time_array)),
                 linestyle='dashed', color='black', linewidth=1)  # plotting RMS
        ax_timeseries.set_xlabel('Time (ns)')
        ax_timeseries.set_ylabel('Amplitude (M^2)')
        ax_timeseries.grid(True)
        ax_timeseries.minorticks_on()
        ax_timeseries.set_title("M_z^2")

    if canvas: canvas.get_tk_widget().pack_forget()  # clear previous canvas

    canvas.draw()
    canvas.get_tk_widget().pack()


def next_ovf():
    total_frames = len(filenames[1:])
    if current_ovf.get() < total_frames - 1:
        current_ovf.set(current_ovf.get() + 1)

    plot_current_ovf()


def prev_ovf():
    if current_ovf.get() > 0:
        current_ovf.set(current_ovf.get() - 1)

    plot_current_ovf()


def load_video():
    cache_state = detect_cache()
    if not cache_state:
        start_cache()
    else:
        load_cache()
        print("Generating animation using multiple OVF files.")
        # note blitting doesn't work for my animation
    anim = ani.FuncAnimation(
        fig,
        update,
        frames=len(filenames[1:]),
        interval=2000
    )

    video_writer = FasterFFMpegWriter(fps=FPS_val.get())

    print("Video saving...")
    anim.save(folder + "/" + vid_name.get(), writer=video_writer)

    print("Video complete!")


def start_load_video():
    # threading necessary so GUI doesn't lock up
    thread = threading.Thread(target=load_video, daemon=True)
    thread.start()


# --- buttons and menu stuff ---
filemenu.add_command(label='Browse Folders', command=open_folder)
filemenu.add_command(label='Cache OVF Files', command=start_cache)
filemenu.add_command(label='Load OVF Cache', command=load_cache)

Imagemenu.add_command(label='Clear Image', command=clear_frame)
Imagemenu.add_separator()
Imagemenu.add_command(label='Display OVF File', command=plot_current_ovf)
# Imagemenu.add_command(label='Plot Core Motion', command=plot_coretracker)
Imagemenu.add_command(label='Load Time Int. Image', command=time_int)

Videomenu.add_command(label='Load Video', command=load_video)

helpmenu.add_command(label="How this works", command=help_popout)

menubar.add_cascade(label="File", menu=filemenu)
menubar.add_cascade(label="Image", menu=Imagemenu)
menubar.add_cascade(label="Video", menu=Videomenu)
menubar.add_cascade(label="Help", menu=helpmenu)

prev_file_button = Button(master=frame, text="prev. file", command=prev_ovf)
prev_file_button.grid(row=0, column=0)

next_file_button = Button(master=frame, text="next file", command=next_ovf)
next_file_button.grid(row=0, column=1)

Mag_comp_Mx_checkbox = Checkbutton(master=frame, text="Mx", variable=mx_var, command=check_mag_comp)
Mag_comp_Mx_checkbox.grid(row=5, column=0)

Mag_comp_My_checkbox = Checkbutton(master=frame, text="My", variable=my_var, command=check_mag_comp)
Mag_comp_My_checkbox.grid(row=6, column=0)

Mag_comp_Mz_checkbox = Checkbutton(master=frame, text="Mz", variable=mz_var, command=check_mag_comp)
Mag_comp_Mz_checkbox.grid(row=7, column=0)

Coretracker_checkbox = Checkbutton(master=frame, text="Coretracker", variable=coretrack_var)
Coretracker_checkbox.grid(row=1, column=1)

# Coretracker_checkbox = Checkbutton(master=frame, text="Track max", variable=coretrack_state_var)
# Coretracker_checkbox.grid(row=2, column=1)

Coretracker_state_button = Button(master=frame, textvariable=coretrack_state_strvar, command=toggle_coretracker)
Coretracker_state_button.grid(row=2, column=1)

log_lim_min_label = Label(master=frame, text="Log limit min")
log_lim_min_label.grid(row=8, column=0)
log_lim_min_entry = Entry(master=frame, textvariable=lim_min)
log_lim_min_entry.focus_force()
log_lim_min_entry.grid(row=9, column=0)

log_lim_max_label = Label(master=frame, text="Log limit max")
log_lim_max_label.grid(row=10, column=0)
log_lim_max_entry = Entry(master=frame, textvariable=lim_max)
log_lim_min_entry.focus_force()
log_lim_max_entry.grid(row=11, column=0)

DPI_label = Label(master=frame, text="DPI (resolution)")
DPI_label.grid(row=12, column=0)
DPI_entry = Entry(master=frame, textvariable=DPI_val)
DPI_entry.focus_force()
DPI_entry.grid(row=13, column=0)

FPS_label = Label(master=frame, text="Video framerate")
FPS_label.grid(row=14, column=0)
FPS_entry = Entry(master=frame, textvariable=FPS_val)
FPS_entry.focus_force()
FPS_entry.grid(row=15, column=0)

vid_name_label = Label(master=frame, text="Video name")
vid_name_label.grid(row=16, column=0)
vid_name_entry = Entry(master=frame, textvariable=vid_name)
vid_name_entry.focus_force()
vid_name_entry.grid(row=17, column=0)

vid_name_label = Label(master=frame, text="Vector Step size")
vid_name_label.grid(row=18, column=0)
vid_name_entry = Entry(master=frame, textvariable=step_val)
vid_name_entry.focus_force()
vid_name_entry.grid(row=19, column=0)



root.config(menu=menubar)
root.mainloop()
