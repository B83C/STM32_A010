import struct
import numpy as np
import matplotlib.pyplot as plt
import os

strm = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Read the binary data from the file
file_path = "training_data.bin"
with open(file_path, "rb") as file:
    data = file.read()

structure_size = 10001
num_structures = len(data) // structure_size

structures = []
marked_for_deletion = set()


def display_image(image, label, ax):
    ax.clear()
    ax.imshow(image, cmap="inferno")
    ax.axis("off")
    ax.set_title(
        f"Label: {strm[label]} {current_index[0]}/{num_structures} {'[DEL]' if current_index[0] in marked_for_deletion else ''}"
    )
    plt.draw()


def on_key(event, structures, current_index, ax):
    if event.key == "left":
        current_index[0] = (current_index[0] - 1) % len(structures)
    elif event.key == "right":
        current_index[0] = (current_index[0] + 1) % len(structures)
    elif event.key == "d":  # Mark or unmark for deletion
        if current_index[0] in marked_for_deletion:
            marked_for_deletion.remove(current_index[0])
        else:
            marked_for_deletion.add(current_index[0])
    elif event.key == "enter":  # Save the remaining entries to the same file
        save_filtered_data(file_path)

    label, image = structures[current_index[0]]
    display_image(image, label, ax)


def save_filtered_data(output_filename):
    with open(output_filename, "wb") as f:
        for i, (label, image) in enumerate(structures):
            if i not in marked_for_deletion:
                f.write(struct.pack("B", label))
                f.write(image.tobytes())
    print(f"Filtered data saved to {output_filename}")

    # Reload the data and update structures and num_structures
    with open(output_filename, "rb") as file:
        new_data = file.read()
    new_num_structures = len(new_data) // structure_size
    new_structures = []
    for i in range(new_num_structures):
        offset = i * structure_size
        new_structures.append(
            [
                new_data[offset],
                np.frombuffer(
                    new_data[offset + 1 : offset + 10001], dtype=np.uint8
                ).reshape((100, 100)),
            ]
        )

    structures[:] = new_structures
    global num_structures
    num_structures = new_num_structures
    current_index[0] = 0

    # Display the first image after reloading
    initial_label, initial_image = structures[current_index[0]]
    display_image(initial_image, initial_label, ax)


for i in range(num_structures):
    offset = i * structure_size
    structures.append(
        [
            data[offset],
            np.frombuffer(data[offset + 1 : offset + 10001], dtype=np.uint8).reshape(
                (100, 100)
            ),
        ]
    )

current_index = [0]
fig, ax = plt.subplots()
fig.canvas.mpl_connect(
    "key_press_event", lambda event: on_key(event, structures, current_index, ax)
)

initial_label, initial_image = structures[current_index[0]]
display_image(initial_image, initial_label, ax)

plt.show()
