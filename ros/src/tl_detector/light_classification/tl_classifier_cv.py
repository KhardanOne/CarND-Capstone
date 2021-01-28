"""
Non-neural-network traffic light detection algorithms.
1.) detect_red_lights_basic(): uses simple pixel counting to find vivid red colors that can be of any shape and size
2.) detect_red_lights_hough(): uses Hough method to identify circle shaped red areas
If both red and green are found, then they both take into account their relative sizes to decide which is more relevant.
"""

import cv2
import numpy as np


def detect_red_lights_basic(image):
  """
  Detects red lights in the image by counting the red and green pixels.
  It only looks for vivid colors.
  Returns True if it finds 10+ vivid red pixels.
  Returns True if the number of vivid green pixels is greater than 
  the number vivid green pixels.
  Returns False otherwise.
  """
  b, g, r  = cv2.split(image)
  bf = b.astype(np.float) / 255.0
  gf = g.astype(np.float) / 255.0
  rf = r.astype(np.float) / 255.0

  rif = rf * (1.0 - bf) * (1.0 - gf) * 2.0 * 256.0
  rif = np.clip(rif, 0.0, 255.0)
  ri8 = rif.astype(np.uint8)
  ri8th = np.zeros_like(ri8)
  rtreshold = 200
  ri8th[ri8 > rtreshold] = 255

  gif = gf * (1.0 - bf) * (1.0 - rf) * 2.0 * 256.0
  gif = np.clip(gif, 0.0, 255.0)
  gi8 = gif.astype(np.uint8)
  gi8th = np.zeros_like(gi8)
  gtreshold = 200
  gi8th[gi8 > gtreshold] = 255

  red_count = np.count_nonzero(ri8th)
  green_count = np.count_nonzero(gi8th)
  print("red:", red_count, "green:", green_count)

  if red_count > 10 and red_count >= 0.8 * green_count:
    return True
  else:
    return False
