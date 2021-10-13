import PIL.Image as Image
import rand_augmentation
import draw_bbox
IMAGE_ROW = 2
IMAGE_COLUMN = 2
IMAGE_SAVE_PATH = r'test.jpg'

# 获取图片集地址下的所有图片名称
image_names = ["../save/228_crop.jpg", "../save/229_crop.jpg", "../save/232_crop.jpg", "../save/256_crop.jpg"]
bbox_names = ["../data/bbox/228_crop.txt", "../data/bbox/229_crop.txt", "../data/bbox/232_crop.txt", "../data/bbox/256_crop.txt"]

dict_ = {}


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


# 定义图像拼接函数
def image_compose(index):
    to_image = Image.new('RGB', (IMAGE_COLUMN * 960, IMAGE_ROW * 540))  # 创建一个新图
    compose_bbox = [] # 合成图片的目标检测框列表

    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            idx = IMAGE_COLUMN * (y - 1) + x - 1 # 放入次序
            src_image = Image.open(image_names[idx]) # 源1/4图片
            src_bbox = load_bbox(bbox_names[idx]) # 源bbox坐标

            randaugment_image, new_bbox, dict_ = rand_augmentation.randaugment(src_image, src_bbox) # 经过randaugment处理图片

            to_image.paste(randaugment_image, ((x - 1) * 960, (y - 1) * 540)) # 添加到对应位置

            # 计算合成图片的目标检测框位置
            if idx == 0:
                pass
            elif idx == 1:
                for i in range(len(new_bbox)):
                    new_bbox[i][0] = new_bbox[i][0] + 960
                    new_bbox[i][2] = new_bbox[i][2] + 960
            elif idx == 2:
                for i in range(len(new_bbox)):
                    new_bbox[i][1] = new_bbox[i][1] + 540
                    new_bbox[i][3] = new_bbox[i][3] + 540
            elif idx == 3:
                for i in range(len(new_bbox)):
                    new_bbox[i][0] = new_bbox[i][0] + 960
                    new_bbox[i][1] = new_bbox[i][1] + 540
                    new_bbox[i][2] = new_bbox[i][2] + 960
                    new_bbox[i][3] = new_bbox[i][3] + 540

            # 加入四分之一图片内经过处理的目标检测框
            for item in new_bbox:
                compose_bbox.append(item)

    draw_bbox.draw_PIL_image(to_image, compose_bbox, index)
    return to_image.save(IMAGE_SAVE_PATH), dict_  # 保存新图

for i in range(25):
    dict_ = image_compose(i)

print(dict_)