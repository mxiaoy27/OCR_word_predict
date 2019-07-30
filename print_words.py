from __future__ import print_function
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import pickle
import argparse
from argparse import RawTextHelpFormatter
import os
import cv2
import random
import numpy as np
import shutil
import traceback
import copy


# 检擦字体文件是否可用
class FontCheck(object):
    def __init__(self, lang_chars, width=32, height=32):  # lang_char是字典
        self.lang_chars = lang_chars
        self.width = width
        self.height = height

    def do(self, font_path):
        width = self.width
        height = self.height
        try:
            for i, char in enumerate(self.lang_chars):
                img = Image.new("RGB", (width, height), "black")
                draw = ImageDraw.Draw(img)

                # 打开并绘制字体
                font = ImageFont.truetype(font_path, int(width * 0.9), )
                draw.text((0, 0), char, (255, 255, 255), font=font)

                data = list(img.getdata())
                sum_val = 0
                for i_data in data:
                    sum_val += sum(i_data)
                if sum_val < 2:
                    return False
        except:
            print("fail to load:%s" % font_path)
            traceback.print_exc(file=sys.stdout)
            return False
        return True


# 生成字体图像
class Font2Image(object):

    def __init__(self, width, height, need_crop, margin):
        self.width = width
        self.height = height
        self.need_crop = need_crop
        self.margin = margin

    def do(self, font_path, char, rotate=0):
        find_image_big_box = FindImageBBox()

        # 黑色背景
        img = Image.new("RGB", (self.width, self.height), "black")
        draw = ImageDraw.Draw(img)

        # 白色文字
        font = ImageFont.truetype(font_path, int(self.width * 1), )
        draw.text((0, 0), char, (255, 255, 255), font=font)

        # 旋转图像增大数据集
        if rotate != 0:
            img = img.rotate(rotate)  # 逆时针旋转

        data = list(img.getdata())  # 每个像素点的RGB值
        sum_val = 0
        for i_data in data:
            sum_val += sum(i_data)  # 整张图片的RGB之和

        if sum_val > 2:  # 存在文字
            np_img = np.asarray(data, dtype='uint8')
            np_img = np_img[:, 0]  # 取所有行的第0个数据 白色是(255,255,255)黑色是(0,0,0)
            np_img = np_img.reshape((self.height, self.width))
            cropped_box = find_image_big_box.do(np_img)  # 找到最小包含矩形
            left, upper, right, lower = cropped_box
            np_img = np_img[upper:lower + 1, left:right + 1]  # 截取图片
            if not self.need_crop:
                preprocess_resize_keep_ratio_fill_bg = PreprocessResizeKeepRatioFillBG(self.width, self.height,
                                                                                       fill_bg=False,
                                                                                       margin=self.margin)
                np_img = preprocess_resize_keep_ratio_fill_bg.do(np_img)
            return np_img
        else:
            # 个别字体的句号太小甚至没有，手动生成句号
            np_img = np.zeros([self.height,self.width,3],np.uint8)
            h = int(self.height*0.8)
            w = int(self.width*0.5)
            
            for j in range(3):
                for i in range(3):
                    np_img[h,w]=255
                    h += 1
                w += 1
                h -= 3
            return np_img


# 对字体图像作等比例缩放
class PreprocessResizeKeepRatio(object):

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def do(self, cv2_img):
        max_width = self.width
        max_height = self.height

        cur_height, cur_width = cv2_img.shape[:2]  # shape[:2] 同时输出高和宽 （切割后的）

        ratio_w = float(max_width) / float(cur_width)
        ratio_h = float(max_height) / float(cur_height)
        ratio = min(ratio_w, ratio_h)

        new_size = (min(int(cur_width * ratio), max_width),
                    min(int(cur_height * ratio), max_height))

        new_size = (max(new_size[0], 1),
                    max(new_size[1], 1),)

        resized_img = cv2.resize(cv2_img, new_size)
        return resized_img


# 查找字体的最小包含矩形
class FindImageBBox(object):

    def __init__(self, ):
        pass

    def do(self, img):
        height = img.shape[0]
        width = img.shape[1]
        v_sum = np.sum(img, axis=0)  # axis=0按列相加，返回每个列的值。
        h_sum = np.sum(img, axis=1)  # axis=1按行的方向相加，返回每个行的值；
        right = width - 1
        low = height - 1
        # 从右往左扫描，遇到非零像素点就以此为字体的右边界
        for i in range(width - 1, -1, -1):
            if v_sum[i] > 0:
                right = i
                break

        # 从下往上扫描，遇到非零像素点就以此为字体的下边界
        for i in range(height - 1, -1, -1):
            if h_sum[i] > 0:
                low = i
                break
        return (0, 0, right, low)


# 把字体图像放到背景图像中
class PreprocessResizeKeepRatioFillBG(object):

    def __init__(self, width, height, fill_bg=False, auto_avoid_fill_bg=True, margin=None):
        self.width = width
        self.height = height
        self.fill_bg = fill_bg
        self.auto_avoid_fill_bg = auto_avoid_fill_bg
        self.margin = margin

    def is_need_fill_bg(cls, cv2_img):  # self是这个类的实例化对象，cls指类本身
        image_shape = cv2_img.shape
        height, width = image_shape
        if height * 3 < width:
            return True
        if width * 3 < height:
            return True
        return False

    # 把字体图像放入背景图像中央
    def put_img_into_center(cls, img_large, img_small):
        width_large = img_large.shape[1]
        height_large = img_large.shape[0]

        width_small = img_small.shape[1]
        height_small = img_small.shape[0]

        if width_large < width_small:
            raise ValueError("width_large <= width_small")
        if height_large < height_small:
            raise ValueError("height_large <= height_small")

        start_width = (width_large - width_small) // 2
        start_height = (height_large - height_small) // 2

        img_large[start_height:start_height + height_small,
        start_width:start_width + width_small] = img_small  # 把img_small放入img_large里
        return img_large

    def do(self, cv2_img):
        # 确定预计字体区域，原图减去边缘长度就是字体的区域
        if self.margin is not None:
            width_minus_margin = max(2, self.width - self.margin)
            height_minus_margin = max(2, self.height - self.margin)
        else:
            width_minus_margin = self.width
            height_minus_margin = self.height

        if len(cv2_img.shape) > 2:
            pix_dim = cv2_img.shape[2]
        else:
            pix_dim = None

        preprocess_resize_keep_ratio = PreprocessResizeKeepRatio(width_minus_margin, height_minus_margin)
        resized_cv2_img = preprocess_resize_keep_ratio.do(cv2_img)

        if self.auto_avoid_fill_bg:
            need_fill_bg = self.is_need_fill_bg(cv2_img)
            if not need_fill_bg:
                self.fill_bg = False
            else:
                self.fill_bg = True

        # 生成最终的字体图像（不含边界）
        if not self.fill_bg:
            ret_img = cv2.resize(resized_cv2_img, (width_minus_margin, height_minus_margin))
        else:
            if pix_dim is not None:
                norm_img = np.zeros((height_minus_margin, width_minus_margin, pix_dim), np.uint8)
            else:
                norm_img = np.zeros((height_minus_margin, width_minus_margin), np.uint8)
            # 将缩放后的字体图像置于背景图像中央
            ret_img = self.put_img_into_center(norm_img, resized_cv2_img)

        # 真正的最终图像！！
        if self.margin is not None:
            if pix_dim is not None:
                norm_img = np.zeros((self.height, self.width, pix_dim), np.uint8)
            else:
                norm_img = np.zeros((self.height, self.width), np.uint8)
            ret_img = self.put_img_into_center(norm_img, ret_img)
        return ret_img


# 数据增强
class dataAugmentation(object):

    def __init__(self, noise=True, dilate=True, erode=True):
        self.noise = noise
        self.dilate = dilate
        self.erode = erode

    # 添加噪声
    def add_noise(cls, img):
        for i in range(5):
            temp_x = np.random.randint(0, img.shape[0])
            temp_y = np.random.randint(0, img.shape[1])
            img[temp_x][temp_y] = 255
        return img

    # 适当腐蚀
    def add_erode(cls, img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.erode(img, kernel)
        return img

    # 适当膨胀
    def add_dilate(cls, img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.dilate(img, kernel)
        return img

    # 做随机扰动
    def do(self, img_list=[]):
        aug_list = copy.deepcopy(img_list)
        for i in range(len(img_list)):
            im = img_list[i]
            # if self.noise and random.random() < 0.5:
            #     im = self.add_noise(im)
            if self.dilate and random.random() < 0.5:
                im = self.add_dilate(im)
            elif self.erode:
                im = self.add_erode(im) 
            aug_list.append(im)
        return aug_list


# 解析参数
def args_parse():
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--out_dir', dest='out_dir', default=None, required=True, help='write a caffe dir')  # 汉字图像的存储目录
    parser.add_argument('--font_dir', dest='font_dir', default=None, required=True,
                        help='font dir to to produce images')  # 放置汉字字体文件的路径
    parser.add_argument('--test_ratio', dest='test_ratio', default=0.2, required=False,
                        help='test dataset size')  # 测试集比例
    parser.add_argument('--width', dest='width', default=None, required=True)  # 生成字体的宽
    parser.add_argument('--height', dest='height', default=None, required=True)
    parser.add_argument('--no_crop', dest='no_crop', default=True, required=False, action='store_true')
    parser.add_argument('--margin', dest='margin', default=0, required=False)  # 边距大小
    parser.add_argument('--rotate', dest='rotate', default=0, required=False, help='max rotate degree 0-45')  # 旋转角度
    parser.add_argument('--rotate_step', dest='rotate_step', default=0, required=False,
                        help='rotate step for the rotate angle')  # 旋转间隔
    parser.add_argument('--need_aug', dest='need_aug', default=False, required=False, help='need data augmentation',
                        action='store_true')
    args = vars(parser.parse_args())  # vars() 函数返回对象的属性和属性值的字典对象。
    return args


# 获取字典
def get_label_dict():
    f = open('./b_label', 'rb')
    label_dict = pickle.load(f)
    f.close()
    return label_dict


if __name__ == "__main__":

    description = '''python print_words.py --out_dir ./dataset --font_dir ./ttf --width 25 --height 30 --margin 3 --rotate 10 --rotate_step 2 --need_aug'''
    options = args_parse()

    out_dir = os.path.expanduser(options['out_dir'])
    font_dir = os.path.expanduser(options['font_dir'])
    test_ratio = float(options['test_ratio'])
    width = int(options['width'])
    height = int(options['height'])
    need_crop = not options['no_crop']
    margin = int(options['margin'])
    rotate = int(options['rotate'])
    need_aug = options['need_aug']
    rotate_step = int(options['rotate_step'])
    train_image_dir_name = "train"
    test_image_dir_name = "test"

    # 将dataset分为train和test两个文件夹分别存储
    train_images_dir = os.path.join(out_dir, train_image_dir_name)  # 把目录和文件名合成一个路径
    test_images_dir = os.path.join(out_dir, test_image_dir_name)

    if os.path.isdir(train_images_dir):  # 判断路径是否存在，若存在则删除重新创建
        shutil.rmtree(train_images_dir)
    os.makedirs(train_images_dir)

    if os.path.isdir(test_images_dir):
        shutil.rmtree(test_images_dir)
    os.makedirs(test_images_dir)

    # 将汉字的label读入，得到（ID：汉字）的映射表label_dict
    lang_chars = get_label_dict()

    for (value, chars) in lang_chars.items():
        print(value, chars)

    font_check = FontCheck(lang_chars)

    if rotate < 0:
        roate = - rotate

    if rotate > 0 and rotate <= 45:
        all_rotate_angles = []  # 记录旋转角度
        for i in range(0, rotate + 1, rotate_step):
            all_rotate_angles.append(i)
        for i in range(-rotate, 0, rotate_step):
            all_rotate_angles.append(i)
        # print(all_rotate_angles)

    # 对于每类字体进行测试
    verified_font_paths = []  # 字体文件目录列表
    # search for file fonts
    for font_name in os.listdir(font_dir):
        path_font_file = os.path.join(font_dir, font_name)
        if font_check.do(path_font_file):
            verified_font_paths.append(path_font_file)

    font2image = Font2Image(width, height, need_crop, margin)

    for (char, value) in lang_chars.items():  # 外层循环是字
        image_list = []
        print(char, value)
        for j, verified_font_path in enumerate(verified_font_paths):  # 内层循环是字体
            if rotate == 0:
                image = font2image.do(verified_font_path, char)
                image_list.append(image)
            else:
                for k in all_rotate_angles:
                    image = font2image.do(verified_font_path, char, rotate=k)
                    image_list.append(image)

        if need_aug:
            data_aug = dataAugmentation()
            image_list = data_aug.do(image_list)

        # 储存
        test_num = len(image_list) * test_ratio
        random.shuffle(image_list)  # 图像列表打乱
        count = 0
        for i in range(len(image_list)):
            img = image_list[i]
            # print(img.shape)
            if count < test_num:
                char_dir = os.path.join(test_images_dir, "%0.2d" % value)
            else:
                char_dir = os.path.join(train_images_dir, "%0.2d" % value)

            if not os.path.isdir(char_dir):
                os.makedirs(char_dir)

            path_image = os.path.join(char_dir, "%d.png" % count)
            cv2.imwrite(path_image, img)
            count += 1
