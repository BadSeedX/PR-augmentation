import PIL
from PIL import ImageDraw
from PIL import Image
import torchvision.transforms.functional as F

def draw_PIL_image(image, boxes, index):
    '''
        Draw PIL image
        image: A PIL image
        labels: A tensor of dimensions (#objects,)
        boxes: A tensor of dimensions (#objects, 4)
    '''

    if type(image) != PIL.Image.Image:
        image = F.to_pil_image(image)
    new_image = image.copy()
    draw = ImageDraw.Draw(new_image)
    # boxes = boxes.tolist()
    for i in range(len(boxes)):
        draw.rectangle(xy=boxes[i],outline="RED")
    new_image.save("../result/img_"+str(index)+".jpg")