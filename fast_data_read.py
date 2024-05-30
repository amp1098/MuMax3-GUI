import numpy as np
import os


direct = os.path.dirname("C:/Users/physinst/Desktop/Mumax/MuMax3_Fall2023/CoreTrackerSample.out/")

# filenames = os.listdir(direct)
#
# print(filenames)


# first_ovf = open(direct + "/m000000.ovf")
#
# for lines in file:
#     print(lines)

class Ovf:
    def __init__(self, directory):
        self.directory = directory

    def reader(self):
        # print("Listing all files in directory...")

        return os.listdir(str(self.directory))

    def data_find(self):
        # print("Isolating OVF files in directory...")
        # this works by checking if the sliced filename has digits in the 1:6 spots, which only ovf files have
        data = []
        for i in Ovf(self.directory).reader():
            if i[1:6].isdigit():
                data.append(i)

        return data

    def header_size_find(self):
        first_ovf = open(self.directory + "/" + str(Ovf(self.directory).data_find()[0]))

        size = 0

        for line in first_ovf:
            if line.startswith("#"):
                size += 1

        return size

    def data_size_find(self):
        first_ovf = open(self.directory + "/" + str(Ovf(self.directory).data_find()[0]))

        size = 0

        for line in first_ovf:
            if not line.startswith("#"):
                size += 1

        return size

    def header_read(self):

        size = Ovf(self.directory).header_size_find()

        line_array = np.full(shape=size, fill_value="0", dtype=object)

        ovf = open(self.directory + "/" + str(Ovf(self.directory).data_find()[0]))

        count = size

        for line in ovf:
            if line.startswith("#"):
                count -= 1
                np.put(line_array, count, line)
        return line_array

    def data_read(self, file_number):

        size = Ovf(self.directory).data_size_find()

        data_array = np.full(shape=size, fill_value="0", dtype=object)

        ovf = open(self.directory + "/" + str(Ovf(self.directory).data_find()[file_number]))

        count = size

        for line in ovf:
            if not line.startswith("#"):
                count -= 1
                np.put(data_array, count, line)
        return data_array

    def data_cache(self):

        files_size = len(Ovf(self.directory).data_find())
        data_size = Ovf(self.directory).data_size_find()

        ovf_array = np.empty((files_size, data_size, 3), dtype=object)

        for file_number in range(files_size):
            print(Ovf(self.directory).data_read(file_number))
            np.put(ovf_array, file_number, Ovf(self.directory).data_read(file_number))

        return ovf_array


thing = Ovf(direct).data_cache()

print(thing)
