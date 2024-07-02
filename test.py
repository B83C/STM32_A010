import argparse
import numpy as np
import matplotlib.pyplot as plt

def display_depth_image(filename):
    # Load pixel data from file
    with open(filename, 'rb') as f:
        pixels = np.frombuffer(f.read(), dtype=np.uint8)

    # Reshape to 100x100 grid
    image = pixels.reshape(100, 100)

    # Display the image with a color gradient
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='inferno')  # Change 'inferno' to any other colormap you prefer
    plt.colorbar()  # Add colorbar to show depth scale
    plt.title('Depth Image')
    plt.xlabel('X Pixels')
    plt.ylabel('Y Pixels')
    plt.show()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Display depth image from file.')
    parser.add_argument('filename', type=str, help='Path to the binary file containing depth data')
    args = parser.parse_args()

    # Call function to display depth image
    display_depth_image(args.filename)
