from __future__ import print_function
from __future__ import division
import os

DEVICE = 'pi'

if DEVICE == 'pi':
    LED_PIN = 18
    LED_FREQ_HZ = 800000
    LED_DMA = 5
    BRIGHTNESS = 255
    LED_INVERT = False
    SOFTWARE_GAMMA_CORRECTION = True

USE_GUI = False

DISPLAY_FPS = True

N_PIXELS = 144

GAMMA_TABLE_PATH = os.path.join(os.path.dirname(__file__), 'gamma_table.npy')

MIC_RATE = 48000

FPS = 50

_max_led_FPS = int(((N_PIXELS * 30e-6) + 50e-6)**-1.0)
assert FPS <= _max_led_FPS, 'FPS must be <= {}'.format(_max_led_FPS)

MIN_FREQUENCY = 200

MAX_FREQUENCY = 12000

N_FFT_BINS = 24

N_ROLLING_HISTORY = 2

MIN_VOLUME_THRESHOLD = 1e-7
