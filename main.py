import tkinter as tk
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import scipy.fftpack as sf
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

# --- creating root window ---
ctypes.windll.shcore.SetProcessDpiAwareness(1)  # this makes all the windows match the device resolution

# root window, contains frames

root = tk.Tk()
root.title("MuMax3 Video Making Suite")
root.geometry('1500x900')

# frame object, this is where the menus and pictures will go

frame = tk.Frame(root)
frame.grid(row=0, column=0, sticky=NW)

root.grid_rowconfigure(0, weight=5)
root.grid_columnconfigure(0, weight=2)

toolbarFrame = tk.Frame(root)
toolbarFrame.grid(row=3, column=1, columnspan=2, sticky=W)

progressbar = ttk.Progressbar()
progressbar.place(x=0, y=530, width=200, height=20)

timer = tk.Label(root, width=30, height=20, text="Total time: ")
timer.place(x=210, y=335)

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

# current showing frame variable, should only go from 1 to length(filenames)
frame_var = tk.IntVar(value=1)

# coretracking toggle variable and list of points for trajectory plotting

coretrack_var = tk.IntVar(value=1)
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


# Other functions for rest of program


def retrieve_header_info():
    load_cache()
    # makes sure that folder is not none so header info can be retrieved
    if folder is None:
        open_folder()

    # reading header info inside OVF files and assigning to list
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
    cmap_top = plt.get_cmap('hsv').copy()
    cmap_top.set_bad(color='k', alpha=0.0)  # transparent if value is 0

    cmap_surf = plt.get_cmap('twilight').copy()
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


# --- Table Reading (For External Field and Time) ---
if folder is not None:
    table = pd.read_csv(folder + '/table.txt', delimiter='\t')

    for col in table.columns:
        col_names.append(col)

data = pd.DataFrame()


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
    fig = plt.figure(figsize=plt.figaspect(0.8,), dpi=150, constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
else:
    if not detect_cache():
        aspect = 1
    else:
        aspect = retrieve_header_info()[1]/retrieve_header_info()[0]
    fig = plt.figure(figsize=plt.figaspect(aspect,)/1.5, dpi=150, constrained_layout=True)
    ax = fig.add_subplot(111)

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


def exp_find(n):
    return np.floor(np.log10(n))


def coretracker(polarity, frame_num):
    # perhaps I could tell this to ignore values to the left or right of a certain x-index?

    # we construct our Mz array from the unpacked ovf file
    Mz = name_dictionary[frame_num]['Mz'].to_numpy(dtype=float)

    # print("Mz: " + str(Mz))
    # we make it a 2D array with dimensions x_size, y_size
    Mz = Mz.reshape(y_size, x_size)
    # we find the max value in this array (or min if the core is down)
    # note that argmax returns a flat index (i.e. it just counts the elements until it find a maximum)
    # so you have to reshape it)

    # we could mask the values that are not in the area we are looking at
    # i just told the program to mask all array elements that are further than halfway across the entire structure

    # mask_arr = np.empty(x_size)
    # for j in np.arange(0, y_size - 1):
    #     masked_row = []
    #     for i in range(x_size):
    #         if i <= 2 * x_size / 3:
    #             element = 1  # mask is on here
    #         else:
    #             element = 0  # mask is off here
    #         masked_row.append(element)
    #     mask_arr = np.vstack([mask_arr, masked_row])
    #
    # Mz = ma.masked_array(Mz, mask_arr)

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


def coretracker_export():
    points = coretracker('up')[1]

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


def toggle_coretracker():
    if coretrack_var.get() == 1:
        coretrack_var.set(0)
    else:
        coretrack_var.set(1)


def fft(time_data, space_data):
    x_int = np.linspace(min(time_data), max(time_data), len(space_data))
    y_int = np.interp(x_int, time_data, space_data)
    sample = len(space_data) / (max(time_data) - min(time_data))  # Sample rate (measurements/second)
    FFT = sf.fft(y_int - y_int.mean())  # calculates FFT, subtract mean to get rid of spike at 0Hz
    f_val = sf.fftfreq(len(y_int)) * sample  # converts time into frequency

    freq_value = f_val[:len(space_data) // 4]  # the 4 here scales the axes into a readable format
    freq_count = np.abs(FFT[:len(space_data) // 4])

    return [freq_value, freq_count]


def update(frames):
    global cmap_surf, cmap_top, fig, ax, filenames, x_size, y_size, z_size, cellx, celly
    temp_filenames = filenames[1:]

    if ThreeD:
        ax.cla()

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

        ax.set_xlabel('x direction', labelpad=-15)
        ax.set_ylabel('y direction', labelpad=-15)
        #'\n' + str(signif(celly * y_size / 10, 5)) + 'um'
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
        # this list stores tracked core locations to produce trajectories for videos

        ax.cla()

        # slower but idk how else to do this
        ax.set(
            xlim=(0, x_size),
            ylim=(0, y_size)
        )

        # scale sometimes needs to be divided by 10, seems to show up if cell size is below 2 digits long

        # ax.set_xlabel('\n' + str(signif(cellx * x_size / 10, 5)) + 'um', labelpad=-15)
        # ax.set_ylabel('\n' + str(signif(celly * y_size / 10, 5)) + 'um', labelpad=-15)

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

        if mx_var.get() == 1:
            ax.pcolormesh(grid_x, grid_y, Mag_x ** 2, cmap=cmap_surf, norm=colors.LogNorm(vmin=min_val, vmax=max_val))
        elif my_var.get() == 1:
            ax.pcolormesh(grid_x, grid_y, Mag_y ** 2, cmap=cmap_surf, norm=colors.LogNorm(vmin=min_val, vmax=max_val))
        elif mz_var.get() == 1:
            ax.pcolormesh(grid_x, grid_y, Mag_z ** 2, cmap=cmap_surf, norm=colors.LogNorm(vmin=min_val, vmax=max_val))
            if coretrack_var.get() == 1:
                core_pos = coretracker('up', filenames[1:][frames])
                coretracker_xlist.append(core_pos[0])
                coretracker_ylist.append(core_pos[1])
                core_x = core_pos[0]
                core_y = core_pos[1]
                ax.scatter(core_x, core_y, s=5, c='white', marker='o')
                ax.plot(coretracker_xlist, coretracker_ylist, c='k', linewidth=0.5)
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

    video_writer = FasterFFMpegWriter(fps=24)
    print("Video saving...")
    anim.save(folder + "/" + vid_name.get(), writer=video_writer)


def start_load_video():
    # threading necessary so GUI doesn't lock up
    thread = threading.Thread(target=load_video, daemon=True)
    thread.start()


def update_canvas(frame_n):
    retrieve_header_info()
    fig.clear()
    ax = fig.add_subplot(111)
    min_val = float(lim_min.get())
    max_val = float(lim_max.get())

    grid_x, grid_y = np.meshgrid(
        np.arange(stop=x_size),
        np.arange(stop=y_size),
    )
    Mx = name_dictionary[frame_n]['Mx'].to_numpy(dtype=float)
    My = name_dictionary[frame_n]['My'].to_numpy(dtype=float)
    Mz = name_dictionary[frame_n]['Mz'].to_numpy(dtype=float)

    Mag_x = np.ma.masked_equal(Mx.reshape(y_size, x_size), 0)
    Mag_y = np.ma.masked_equal(My.reshape(y_size, x_size), 0)
    Mag_z = np.ma.masked_equal(Mz.reshape(y_size, x_size), 0)

    set_colormaps()

    if mx_var.get() == 1:
        ax.pcolormesh(grid_x, grid_y, Mag_x ** 2, cmap=cmap_surf, norm=colors.LogNorm(vmin=min_val, vmax=max_val))
    elif my_var.get() == 1:
        ax.pcolormesh(grid_x, grid_y, Mag_y ** 2, cmap=cmap_surf, norm=colors.LogNorm(vmin=min_val, vmax=max_val))
    elif mz_var.get() == 1:
        ax.pcolormesh(grid_x, grid_y, Mag_z ** 2, cmap=cmap_surf, norm=colors.LogNorm(vmin=min_val, vmax=max_val))
        if coretrack_var.get() == 1:
            core_pos = coretracker('up', frame_n)
            core_x = core_pos[0]
            core_y = core_pos[1]
            ax.scatter(core_x, core_y, s=5, c='black')
        #fig.colorbar(cs)


    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0)


def prev_frame():
    # display previous frame
    # count frame_var down 1

    if frame_var.get() > 1:
        frame_var.set(frame_var.get() - 1)

    update_canvas(filenames[frame_var.get()])


def next_frame():
    # display next frame
    # count frame_var up 1

    if frame_var.get() < len(filenames):
        frame_var.set(frame_var.get() + 1)

    update_canvas(filenames[frame_var.get()])


def time_int():
    retrieve_header_info()
    fig.clear()
    ax = fig.add_subplot(111)
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

    # ax.set_xlabel('\n' + str(signif(cellx * x_size, 5)) + 'um', labelpad=-15)
    # ax.set_ylabel('\n' + str(signif(celly * y_size, 5)) + 'um', labelpad=-15)

    min_val = float(lim_min.get())
    max_val = float(lim_max.get())

    ax.pcolormesh(grid_x, grid_y, sum_total ** 2, cmap=cmap_surf, norm=colors.LogNorm(
        vmin=min_val, vmax=max_val)
                  )

    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0)


# --- buttons ---


browse_file_button = Button(master=frame, text="Browse Folders", command=open_folder)
browse_file_button.grid(row=0, column=0)

start_cache_button = Button(master=frame, text="Cache OVF Files", command=start_cache)
start_cache_button.grid(row=1, column=0)

load_cache_button = Button(master=frame, text="Load OVF Cache", command=load_cache)
load_cache_button.grid(row=2, column=0)

video_button = Button(master=frame, text="Load Video", command=start_load_video)
video_button.grid(row=3, column=0)

time_int_button = Button(master=frame, text="Load Time Integration Image", command=time_int)
time_int_button.grid(row=4, column=0)

Mag_comp_Mx_checkbox = Checkbutton(master=frame, text="Mx", variable=mx_var, command=check_mag_comp)
Mag_comp_Mx_checkbox.grid(row=5, column=0)

Mag_comp_My_checkbox = Checkbutton(master=frame, text="My", variable=my_var, command=check_mag_comp)
Mag_comp_My_checkbox.grid(row=6, column=0)

Mag_comp_Mz_checkbox = Checkbutton(master=frame, text="Mz", variable=mz_var, command=check_mag_comp)
Mag_comp_Mz_checkbox.grid(row=7, column=0)

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

vid_name_label = Label(master=frame, text="Video name")
vid_name_label.grid(row=12, column=0)
vid_name_entry = Entry(master=frame, textvariable=vid_name)
vid_name_entry.focus_force()
vid_name_entry.grid(row=13, column=0)

coretracker_checkbox = Checkbutton(master=frame, text="Coretracking", variable=coretrack_var, command=toggle_coretracker)
coretracker_checkbox.grid(row=14, column=0)

prev_frame_button = Button(master=frame, text="Prev. Frame", command=prev_frame)
prev_frame_button.grid(row=15, column=0)
prev_frame_button.bind('<Left>', prev_frame)

next_frame_button = Button(master=frame, text="Next. Frame", command=next_frame)
next_frame_button.grid(row=16, column=0)
next_frame_button.bind('<Right>', prev_frame)

root.mainloop()
