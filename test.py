import serial
import serial.tools.list_ports
import time
import pygame
import struct
import os

# Constants
FRAME_BEGIN_FLAG = 0xFF00
FRAME_END_FLAG = 0xDD
FRAME_HEAD_SIZE = 20
WIDTH = 100
HEIGHT = 100
SERIAL_PORT_BAUDRATE = 115200

# Gradient colors (7 colors, will be interpolated to 256)
GRADIENT_COLORS = [
    (127, 0, 0),  # Dark red
    (255, 0, 0),  # Red
    (255, 255, 0),  # Yellow
    (0, 255, 0),  # Green
    (0, 255, 255),  # Cyan
    (0, 0, 255),  # Blue
    (0, 0, 76),  # Dark blue
]


# Function to create a gradient
def create_gradient(colors, size=256):
    gradient = []
    for i in range(size):
        f = i / (size - 1) * (len(colors) - 1)
        i1, i2 = int(f), min(int(f) + 1, len(colors) - 1)
        r1, g1, b1 = colors[i1]
        r2, g2, b2 = colors[i2]
        t = f - i1
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        gradient.append((r, g, b))
    return gradient


# Initialize gradient
gradient = create_gradient(GRADIENT_COLORS)


# Find the serial port
def find_serial_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if port.product and "SIPEED Meta Sense Lite" in port.product:
            return port.device
    raise Exception("SIPEED Meta Sense Lite not found")


# Initialize the serial port
def init_serial():
    port_name = find_serial_port()
    ser = serial.Serial(port_name, SERIAL_PORT_BAUDRATE, timeout=0.01)
    time.sleep(0.02)
    ser.write(b"AT+DISP=2\r")
    time.sleep(0.02)
    ser.write(b"AT+FPS=19\r")
    time.sleep(0.02)
    ser.write(b"AT+UNIT=1\r")
    time.sleep(0.02)
    ser.write(b"AT+ANTIMMI=-1\r")
    time.sleep(0.02)
    return ser


# Main function
def main():
    ser = init_serial()
    buf = bytearray(10022)
    output = [(0, 0, 0)] * (WIDTH * HEIGHT)

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH * 4, HEIGHT * 4))
    pygame.display.set_caption("Depth Image")

    # Open file for writing
    if not os.path.exists("training_data.bin"):
        open("training_data.bin", "wb").close()
    file = open("training_data.bin", "ab")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                key = event.key
                if pygame.key.get_mods() & pygame.KMOD_ALT:
                    if key >= pygame.K_0 and key <= pygame.K_9:
                        unit = key - pygame.K_0 + 1
                        ser.write(f"AT+UNIT={unit}\r".encode())
                elif key == pygame.K_BACKSPACE:
                    if file.seek(-10001, os.SEEK_END) != -1:
                        file.truncate()
                        print("Deleted last label")
                else:
                    print(f"Saved label {key}")
                    file.write(struct.pack("B", key))
                    file.write(buf[20 : 20 + WIDTH * HEIGHT])

        if ser.readinto(buf) == len(buf):
            header = struct.unpack_from("<HHBBB4sBBBHHB", buf)
            if header[0] == FRAME_BEGIN_FLAG:
                print(f"Frame id: {header[10]}")
                for i, pixel in enumerate(buf[20 : 20 + WIDTH * HEIGHT]):
                    output[i] = gradient[pixel]

                pygame.surfarray.blit_array(screen, output)
                pygame.display.flip()
        time.sleep(0.01)

    file.close()
    ser.close()
    pygame.quit()


if __name__ == "__main__":
    main()
