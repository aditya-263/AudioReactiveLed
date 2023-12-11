#!/usr/bin/env python3

import time
from rpi_ws281x import *
import argparse
import config

LED_COUNT = config.N_PIXELS 
LED_PIN = config.LED_PIN  
LED_FREQ_HZ = config.LED_FREQ_HZ   
LED_DMA = config.LED_DMA  
LED_BRIGHTNESS = config.BRIGHTNESS  
LED_INVERT = config.LED_INVERT  

LED_CHANNEL = 0  


def color_wipe(strip, color, wait_ms=50):
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, color)
        strip.show()
        time.sleep(wait_ms / 1000.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clear', action='store_true', help='clear the display on exit')
    args = parser.parse_args()

    strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
    strip.begin()

    color_wipe(strip, Color(0, 0, 0), 10)
