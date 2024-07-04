import struct
import numpy as np
import matplotlib.pyplot as plt
import os 

class Data:
    def __init__(self, label, pixels):
        self.label = label
        self.pixels = pixels

def decode_data(binary_data):
    label = struct.unpack('B', binary_data[:1])[0]
    pixels = binary_data[1:10001]
    # pixels = (struct.unpack('10000B', binary_data[1:]))
    return Data(label, pixels)


strm = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# Read the binary data from the file
with open('training_data.bin', 'rb') as file:
    data = file.read()

structure_size = 10001
num_structures = len(data) // structure_size

structures = []

def display_image(image, label, ax):
    ax.clear()
    ax.imshow(image, cmap='inferno')
    ax.axis('off')
    ax.set_title(f'Label: {strm[label]}')
    plt.draw()

def on_key(event, structures, current_index, ax):
    if event.key == 'left':
        current_index[0] = (current_index[0] - 1) % len(structures)
    elif event.key == 'right':
        current_index[0] = (current_index[0] + 1) % len(structures)

    label, image = structures[current_index[0]]
    display_image(image, label, ax)

for i in range(num_structures):
    offset = i * structure_size
    structures.append([data[offset], np.frombuffer(data[offset +1:offset + 10001], dtype=np.uint8).reshape((100,100))])


current_index = [0]
fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', lambda event: on_key(event, structures, current_index, ax))

initial_label, initial_image = structures[current_index[0]]
display_image(initial_image, initial_label, ax)

plt.show()
