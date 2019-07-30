import sys
import cv2
import numpy as np

img = cv2.imread("2.png", 1)   # 读取图片，第二个参数必须为1
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化为灰度图

img_thre = img_gray
cv2.threshold(img_gray, 225, 255, cv2.THRESH_BINARY_INV, img_thre)  # 把灰度图转化成二值图
cv2.imwrite("./row_cut/thre_res.png", img_thre)

row_white = []   # 记录图片每一行的白色像素
row_black = []   # 黑色
height = img_thre.shape[0]  # 图片的高
width = img_thre.shape[1]   # 宽
row_white_max = 0
row_black_max = 0
for i in range(height):
    w = 0  # 这一行白色像素的总数
    b = 0  # 黑色
    for j in range(width):
        if img_thre[i][j] == 255:
            w += 1
        if img_thre[i][j] == 0:
            b += 1
    row_white_max = max(row_white_max, w)  # 获取图片行或列的最大白色像素数
    row_black_max = max(row_black_max, b)  # 黑色   一般情况下就是图片的高
    row_white.append(w)
    row_black.append(b)


# 找到每个字母的切割终点
def find_end(start, direct, black_max, n, black=[]):
    end = start+1
    for m in range(start+1, direct):
        if (black[m]) > (n * black_max):
            end = m
            break
    return end


# 按行分割图片
img_id = []   # 按行分割的图片数组
n = 0
row = -1
row_start = 0
row_end = 2
while row < height-1:
    row += 1
    if (row_white[row]) > (0.0001 * row_white_max):
        row_start = row
        row_end = find_end(row_start, height, row_black_max, 0.9999, row_black)
        row = row_end
        if row_end-row_start > 5:
            cj = img[row_start:row_end, 0:width]
            cv2.imwrite("./row_cut/"+str(n)+".png", cj)
            img_id.append(n)
            n += 1


# 最后处理切割出的图片比例以及边缘 主要是对于过瘦字体和标点符号的优化
def final_process(img, img_height, img_width):  # 切割出的字母和高和宽
    if((img_height * 0.7) > img_width):
        final_img = np.zeros([img_height+3, int(img_height*0.7)+4, 3], np.uint8)
        width_start = int((int(img_height*0.7) + 4 - img_width)/2)
        final_img[3:img_height+3, width_start:width_start+img_width] = img
        return final_img
    else:
        final_img = np.zeros([img_height+3, img_width+4, 3], np.uint8)
        final_img[3:img_height+3, 2:img_width+2] = img
        return final_img


# 分割每一行
num = len(img_id)
n_ = 0  # 记录切割次数
for p in range(num):  # 行循环
    img_row = cv2.imread("./row_cut/" + str(p) + ".png", 1)
    img_gray = cv2.cvtColor(img_row, cv2.COLOR_BGR2GRAY)
    img_row_thre = img_gray
    cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV, img_row_thre)
    cv2.imwrite("./row_cut/" + str(p) + "_res.png", img_row_thre)

    col_white = []
    col_black = []
    height = img_row_thre.shape[0]
    width = img_row_thre.shape[1]
    col_white_max = 0
    col_black_max = 0
    for i in range(width):
        w = 0  # 这一行白色像素的总数
        b = 0  # 黑色
        for j in range(height):
            if img_row_thre[j][i] == 255:
                w += 1
            if img_row_thre[j][i] == 0:
                b += 1
        col_white_max = max(col_white_max, w)
        col_black_max = max(col_black_max, b)
        col_white.append(w)
        col_black.append(b)

    last_col = 0  # 记录上一次切割的位置
    new_col = 0  # 扫描完空格后的位置
    word_width = 0
    col = 0  # 扫描的列
    col_start = 1
    col_end = 2
    while (col < width - 1):
        col += 1
        if (col_white[col]) > (0.02 * col_white_max):  # 如果这一列的白色大于最大列的白的像素数的0.02，则认为这一列有字母
            col_start = col
            col_end = find_end(col_start, width, col_black_max, 0.98, col_black)
            col = col_end
            word_width = col_end - col_start
            if (word_width > 3) or ((col_white[col_start]) > (0.5 * col_white_max)):
                cj = img_row[0:height, col_start:col_end]
                cj = cv2.bitwise_not(cj)  # 像素反取
                img = final_process(cj, height, col_end - col_start)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                img = cv2.dilate(img, kernel)
                cv2.imwrite("./image_temp/" + "%0.4d" % n_ + ".png", img)  # 命名规则为“分割次数”
                n_ += 1
            last_col = col
            new_col = col
        else:  # 判断空格
            while True:
                new_col += 1
                if (new_col > (width - 1)) or (col_white[new_col] > (0.02 * col_white_max)):  # 扫描到下一处字体或图片结束
                    break
            col = new_col - 1
            if (col - last_col) > (height * 0.4):  # 生成黑色方块，当作空格
                create_space = np.zeros([height, int(height * 0.7), 1],
                                        np.uint8)  # 用numpy生成单通道二维数组，height行，height列，单通道，8位整形
                create_space[:, :, 0] = np.ones([height, int(height * 0.7)]) * 0  # 单通道每个元素为1, 乘以0  得到黑色方块
                cv2.imwrite("./image_temp/" + "%0.4d" % n_ + ".png", create_space)
                n_ += 1
