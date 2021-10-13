import PIL
import torch
import numpy as np
from PIL import ImageDraw
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms.functional as TF

def draw_PIL_image(image, boxes):
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

    new_image.save("test_.jpg")

def rotate(img, bbox, angle = 45):

    bboxes = np.zeros((len(bbox), 4), dtype=int)
    for i in range(len(bbox)):
        for j in range(4):
            bboxes[i][j] = int(bbox[i][j])

    # 翻转图片
    rotate_img = img.rotate(angle, expand = True)

    # 获取原图片长宽
    w = img.width
    h = img.height
    cx = w / 2
    cy = h / 2

    angle = np.radians(angle)
    alpha = np.cos(angle)
    beta = np.sin(angle)
    AffineMatrix = torch.tensor([[alpha, beta, (1 - alpha) * cx - beta * cy],
                             [-beta, alpha, beta * cx + (1 - alpha) * cy]])

    # 获取目标检测框长和宽并转化为张量
    box_width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    box_height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    # 获取目标检测框的横纵坐标列表
    x1 = torch.tensor(bboxes[:, 0].reshape(-1, 1))
    y1 = torch.tensor(bboxes[:, 1].reshape(-1, 1))
    x2 = x1 + torch.tensor(box_width)
    y2 = y1
    x3 = x1
    y3 = y1 + torch.tensor(box_height)
    x4 = torch.tensor(bboxes[:, 2].reshape(-1, 1))
    y4 = torch.tensor(bboxes[:, 3].reshape(-1, 1))

    corners = torch.stack((x1, y1, x2, y2, x3, y3, x4, y4), dim=1)
    corners.reshape(len(bbox), 8)
    corners = corners.reshape(-1, 2)
    corners = torch.cat((corners.float(), torch.ones(corners.shape[0], 1)), dim=1)

    cos = np.abs(AffineMatrix[0, 0])
    sin = np.abs(AffineMatrix[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    AffineMatrix[0, 2] += (nW / 2) - cx
    AffineMatrix[1, 2] += (nH / 2) - cy

    rotate_corners = torch.mm(AffineMatrix, corners.t()).t()
    rotate_corners = rotate_corners.reshape(-1, 8)

    x_corners = rotate_corners[:, [0, 2, 4, 6]]
    y_corners = rotate_corners[:, [1, 3, 5, 7]]

    x_min, _ = torch.min(x_corners, dim=1)
    x_min = x_min.reshape(-1, 1)
    y_min, _ = torch.min(y_corners, dim=1)
    y_min = y_min.reshape(-1, 1)
    x_max, _ = torch.max(x_corners, dim=1)
    x_max = x_max.reshape(-1, 1)
    y_max, _ = torch.max(y_corners, dim=1)
    y_max = y_max.reshape(-1, 1)

    new_bbox_contents = torch.cat((x_min, y_min, x_max, y_max), dim=1)

    scale_x = rotate_img.width / w
    scale_y = rotate_img.height / h

    # 将翻转后的图片转为原大小
    rotate_img = rotate_img.resize((w, h))
    # 保证检测框在图片中
    new_bbox_contents /= torch.Tensor([scale_x, scale_y, scale_x, scale_y])
    new_bbox_contents[:, 0] = torch.clamp(new_bbox_contents[:, 0], 0, w)
    new_bbox_contents[:, 1] = torch.clamp(new_bbox_contents[:, 1], 0, h)
    new_bbox_contents[:, 2] = torch.clamp(new_bbox_contents[:, 2], 0, w)
    new_bbox_contents[:, 3] = torch.clamp(new_bbox_contents[:, 3], 0, h)

    new_bbox = new_bbox_contents.numpy().tolist()
    print(new_bbox)
    # draw_PIL_image(rotate_img, new_bbox_contents)
    return rotate_img

# 读取图片对应的bbox坐标
def load_bbox(path):
    with open(path, "r") as f:
        contents = f.readlines()
        bbox = []
        for _, content in enumerate(contents):
            cls, x1, y1, x2, y2 = content.split()
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            bbox.append([x1,y1,x2,y2])
    return bbox       # bbox [[x_min,y_min,x_max,y_max],..]


img = Image.open("../data/img/228.jpg")
bbox = load_bbox("../data/bbox/228.txt")
# bbox = [[348, 409, 568], [174, 229, 282], [415, 524,614.], [246,340,345]]
new_img = rotate(img, bbox)
plt.figure("test")
plt.imshow(new_img)
plt.show()