import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import transform

transforms = [
    "AdjustContrast", "AdjustBrightness","AdjustSaturation",
    "AdjustHue", "AdjustGamma", "LightingNoise", "Hflip", "Vflip","Rotate"
]

NAME_TO_FUNC = {
    "AdjustContrast": transform.adjustContrast,
    "AdjustBrightness": transform.adjustBrightness,
    "AdjustSaturation": transform.adjustSaturation,
    "AdjustHue": transform.adjustHue,
    "AdjustGamma": transform.adjustGamma,
    "LightingNoise": transform.lightingNoise,
    "Hflip": transform.hflip,
    "Vflip": transform.vflip,
    "Rotate": transform.rotate,
}
dict = {}
def randaugment(img, bbox, N = 1):
    ops = np.random.choice(transforms, N)
    transform_img = img
    new_bbox = bbox
    for op in ops:
        print(op)
        if dict.get(op) == None:
            dict[op] = 1
        else:
            dict[op] += 1
        transfrom_img, new_bbox= NAME_TO_FUNC[op](transform_img, new_bbox)
    return transfrom_img, new_bbox, dict

"""
img = Image.open("../save/256_crop.jpg")
plt.figure("test")
plt.imshow(randaugment(img))
plt.show()
"""




