# MuMax3-GUI
This program allows for users to analyze and animate OVF files outputted by MuMax3.

Currently this code allows for users to convert any number of OVF files (OVFs) into videos and images. The images can be of each OVF file directly, 
or they can be a sum of all files in a folder to show any repeated dynamical behaviors. The videos use each OVF file in a folder titled "m[number]"
to create a .mp4 video via FFMPEG.

To save time with repeated calculations, the program also caches OVF files into a pickled python dictionary file
whose keys are the file names and whose items are the data from the OVF file. This is what is itereated over in the Update function.

Tracking of minima and maxima is supported to allow for magnetic vortex core trajectory mapping.

The GUI currenly has a single canvas element to visualize OVF files and the DPI function is bugged.
