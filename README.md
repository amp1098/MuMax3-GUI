# MuMax3-GUI
This program allows for users to analyze and animate OVF files outputted by MuMax3.

Currently this code allows for users to convert any number of OVF files (OVFs) into both time-integrated images and videos.
To save time with repeated calculations, the program also caches OVF files into a pickled python dictionary file
whose keys are the file names and whose items are the data from the OVF file. This is what is itereated over in the Update function.

Tracking of minima and maxima is supported to allow for magnetic vortex core trajectory mapping.
