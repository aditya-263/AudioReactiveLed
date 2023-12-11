from __future__ import print_function
from __future__ import division

import platform
import numpy as np
import config

if config.DEVICE == 'esp8266':
    import socket
    _sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
elif config.DEVICE == 'pi':
    from rpi_ws281x import *
    strip = Adafruit_NeoPixel(config.N_PIXELS, config.LED_PIN,
                                       config.LED_FREQ_HZ, config.LED_DMA,
                                       config.LED_INVERT, config.BRIGHTNESS)
    strip.begin()
elif config.DEVICE == 'blinkstick':
    from blinkstick import blinkstick
    import signal
    import sys
    def signal_handler(signal, frame):
        all_off = [0]*(config.N_PIXELS*3)
        stick.set_led_data(0, all_off)
        sys.exit(0)

    stick = blinkstick.find_first()
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

_gamma = np.load(config.GAMMA_TABLE_PATH)

_prev_pixels = np.tile(253, (3, config.N_PIXELS))

pixels = np.tile(1, (3, config.N_PIXELS))

_is_python_2 = int(platform.python_version_tuple()[0]) == 2

def _update_esp8266():
    global pixels, _prev_pixels
    pixels = np.clip(pixels, 0, 255).astype(int)
    p = _gamma[pixels] if config.SOFTWARE_GAMMA_CORRECTION else np.copy(pixels)
    MAX_PIXELS_PER_PACKET = 126
    idx = range(pixels.shape[1])
    idx = [i for i in idx if not np.array_equal(p[:, i], _prev_pixels[:, i])]
    n_packets = len(idx) // MAX_PIXELS_PER_PACKET + 1
    idx = np.array_split(idx, n_packets)
    for packet_indices in idx:
        m = '' if _is_python_2 else []
        for i in packet_indices:
            if _is_python_2:
                m += chr(i) + chr(p[0][i]) + chr(p[1][i]) + chr(p[2][i])
            else:
                m.append(i)  # Index of pixel to change
                m.append(p[0][i])  # Pixel red value
                m.append(p[1][i])  # Pixel green value
                m.append(p[2][i])  # Pixel blue value
        m = m if _is_python_2 else bytes(m)
        _sock.sendto(m, (config.UDP_IP, config.UDP_PORT))
    _prev_pixels = np.copy(p)


def _update_pi():
    global pixels, _prev_pixels
    pixels = np.clip(pixels, 0, 255).astype(int)
    p = _gamma[pixels] if config.SOFTWARE_GAMMA_CORRECTION else np.copy(pixels)
    r = np.left_shift(p[0][:].astype(int), 8)
    g = np.left_shift(p[1][:].astype(int), 16)
    b = p[2][:].astype(int)
    rgb = np.bitwise_or(np.bitwise_or(r, g), b)
    for i in range(config.N_PIXELS):
        if np.array_equal(p[:, i], _prev_pixels[:, i]):
            continue
            
        strip._led_data[i] = int(rgb[i])
    _prev_pixels = np.copy(p)
    strip.show()

def _update_blinkstick():
    global pixels
    
    pixels = np.clip(pixels, 0, 255).astype(int)
    p = _gamma[pixels] if config.SOFTWARE_GAMMA_CORRECTION else np.copy(pixels)
    r = p[0][:].astype(int)
    g = p[1][:].astype(int)
    b = p[2][:].astype(int)

    newstrip = [None]*(config.N_PIXELS*3)

    for i in range(config.N_PIXELS):
        newstrip[i*3] = g[i]
        newstrip[i*3+1] = r[i]
        newstrip[i*3+2] = b[i]
    stick.set_led_data(0, newstrip)


def update():
    if config.DEVICE == 'esp8266':
        _update_esp8266()
    elif config.DEVICE == 'pi':
        _update_pi()
    elif config.DEVICE == 'blinkstick':
        _update_blinkstick()
    else:
        raise ValueError('Invalid device selected')

if __name__ == '__main__':
    import time
    pixels *= 0
    pixels[0, 0] = 255  # Set 1st pixel red
    pixels[1, 1] = 255  # Set 2nd pixel green
    pixels[2, 2] = 255  # Set 3rd pixel blue
    print('Starting LED strand test')
    while True:
        pixels = np.roll(pixels, 1, axis=1)
        update()
        time.sleep(.1)
