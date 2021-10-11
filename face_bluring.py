from facenet_pytorch import MTCNN
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
from IPython import display
from PIL import Image, ImageDraw, ImageFilter
import numpy as np


def make_ellipse_mask(size, x0, y0, x1, y1, blur_radius):
    img = Image.new("L", size, color=0)
    draw = ImageDraw.Draw(img)
    draw.ellipse((x0, y0, x1, y1), fill=255)
    return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device)

# video = mmcv.VideoReader('video.mp4')
# frames_tracked = []
# frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
# img = mmcv.imread('test.bmp') # 1232*1024
name = '871.jpg'
img = mmcv.imread(name)

# img = mmcv.imrotate(img, 90)
# print(img.show())
# mmcv.imresize_like(img, frames[0], return_scale=False)
# print(img)
img = mmcv.imresize(img, (1920, 1080))
# mmcv.imresize(img, (1920, 1080), return_scale=False)
# print(img)
img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# print(img)
# print(frames[0])
boxes, _ = mtcnn.detect(img)
print(boxes[0][0])


def scale_cords_scalefill(scaled_img_shape, coords, org_img_shape):
    hr = scaled_img_shape[0] / org_img_shape[1]
    wr = scaled_img_shape[1] / org_img_shape[0]

    print('hr ' + str(hr))
    print('wr ' + str(wr))

    # coords[:, [0, 1]] /= wr
    # coords[:, [1, 0]] /=hr
    # , coords[1][0]
    # , coords[1][1]
    coords[0][1] /= wr  # y0
    coords[1][1] /= wr  # y1

    coords[0][0] /= hr  # xo
    coords[1][0] /= hr  # x1
    return coords


# display.Video('video.mp4', width=640)
frame_draw = img.copy()
draw = ImageDraw.Draw(frame_draw)
# box2 = []
for box in boxes:
    # draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
    # draw.ellipse((box[0], box[1], box[2], box[3]), fill=0)
    print(box)
    img2 = mmcv.imread(name)
    box2 = np.array(box)
    box2 = box2.reshape(2, 2)
    print(box2)
    print(img2.shape)
    print(frame_draw.size)
    cropped_image = frame_draw.crop((box[0], box[1], box[2], box[3]))
    blurred_image = cropped_image.filter(ImageFilter.GaussianBlur(radius=20))
    # draw.filter(ImageFilter.GaussianBlur(radius=5))
    # frame_draw = make_ellipse_mask(frame_draw.size, box[0], box[1], box[2], box[3], 5)
    # print(box[0])
    # draw.filter(ImageFilter.BoxBlur(5))

# Add to frame list
# frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
# d = display.display(frames_tracked[0], display_id=True)
print(cropped_image.show())
cropped_image.show()
d = display.display(cropped_image, display_id=True)
d = display.display(blurred_image, display_id=True)
# print(frames_tracked[0].show())
# frame_draw.show()
# img = mmcv.imresize(frame_draw, (640, 360))

values = scale_cords_scalefill(frame_draw.size, box2, img2.shape)
print("the values are" + str(values))
img4 = Image.open(name)
# d = display.display(img4, display_id=True)
# img4.show()
img4.size
# 3581.3015 5069.2354 3611.5146 5189.422
img3 = img4.crop((int(values[0][0]), int(values[0][1]), int(values[1][0]), int(values[1][1])))
# img3 = img4.crop((7134, 2505, 7257, 2614))
img5 = img4.crop((int(values[0][0]), int(values[0][1]), int(values[1][0]), int(values[1][1])))
print(values[0][0], values[0][1], values[1][0], values[1][1])
# d = display.display(img3, display_id=True)
d = display.display(img5, display_id=True)
blurrred_image = img3.filter(ImageFilter.GaussianBlur(radius=20))
d = display.display(blurrred_image, display_id=True)
img4.paste(blurrred_image,
           (int(values[0][0]), int(values[0][1]), int(values[1][0]), int(values[1][1])))  # (7134, 2505, 7257, 2614)
d = display.display(img4, display_id=True)
# img3.show()
img4.save("new_ImageDraw_blurred.jpg")
print('done')

# kitten_image =frame_draw.copy()    #Image.open("download.png")
# overlay_image = Image.new("RGB", kitten_image.size,
# color="orange")  # This could be a bitmap fill too, but let's just make it orange
# mask_image = make_ellipse_mask(kitten_image.size, 150, 70, 350, 250, 5)
# masked_image = Image.composite(overlay_image, kitten_image, mask_image)

# masked_image.show()


# frame_draw = frame_draw.resize((8192, 4096), Image.BILINEAR)
# frame_draw.save("ImageDraw4.jpg")