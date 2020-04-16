from PIL import Image
import pyqrcode
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import os

# 动森支持的rgb色值表，144种彩色+15种灰度
color_dict = {(255, 239, 255): '00', (255, 154, 173): '01', (239, 85, 156): '02', (255, 101, 173): '03',
              (255, 0, 99): '04', (189, 69, 115): '05', (206, 0, 82): '06', (156, 0, 49): '07', (82, 32, 49): '08',
              (255, 186, 206): '10', (255, 117, 115): '11', (222, 48, 16): '12', (255, 85, 66): '13', (255, 0, 0): '14',
              (206, 101, 99): '15', (189, 69, 66): '16', (189, 0, 0): '17', (140, 32, 33): '18', (222, 207, 189): '20',
              (255, 207, 99): '21', (222, 101, 33): '22', (255, 170, 33): '23', (255, 101, 0): '24', (189, 138, 82): '25',
              (222, 69, 0): '26', (189, 69, 0): '27', (99, 48, 16): '28', (255, 239, 222): '30', (255, 223, 206): '31',
              (255, 207, 173): '32', (255, 186, 140): '33', (255, 170, 140): '34', (222, 138, 99): '35',
              (189, 101, 66): '36', (156, 85, 49): '37', (140, 69, 33): '38', (255, 207, 255): '40',
              (239, 138, 255): '41', (206, 101, 222): '42', (189, 138, 206): '43', (206, 0, 255): '44',
              (156, 101, 156): '45', (140, 0, 173): '46', (82, 0, 115): '47', (49, 0, 66): '48', (255, 186, 255): '50',
              (255, 154, 255): '51', (222, 32, 189): '52', (255, 85, 239): '53', (255, 0, 206): '54',
              (140, 85, 115): '55', (189, 0, 156): '56', (140, 0, 99): '57', (82, 0, 66): '58',
              (222, 186, 156): '60', (206, 170, 115): '61', (115, 69, 49): '62', (173, 117, 66): '63',
              (156, 48, 0): '64', (115, 48, 33): '65', (82, 32, 0): '66', (49, 16, 0): '67', (33, 16, 0): '68',
              (255, 255, 206): '70', (255, 255, 115): '71', (222, 223, 33): '72', (255, 255, 0): '73',
              (255, 223, 0): '74', (206, 170, 0): '75', (156, 154, 0): '76', (140, 117, 0): '77', (82, 85, 0): '78',
              (222, 186, 255): '80', (189, 154, 239): '81', (99, 48, 206): '82', (156, 85, 255): '83', (99, 0, 255): '84',
              (82, 69, 140): '85', (66, 0, 156): '86', (33, 0, 99): '87', (33, 16, 49): '88', (189, 186, 255): '90',
              (140, 154, 255): '91', (49, 48, 173): '92', (49, 85, 239): '93', (0, 0, 255): '94', (49, 48, 140): '95',
              (0, 0, 173): '96', (16, 16, 99): '97', (0, 0, 33): '98', (156, 239, 189): 'a0', (99, 207, 115): 'a1',
              (33, 101, 16): 'a2', (66, 170, 49): 'a3', (0, 138, 49): 'a4', (82, 117, 82): 'a5', (33, 85, 0): 'a6',
              (16, 48, 33): 'a7', (0, 32, 16): 'a8', (222, 255, 189): 'b0', (206, 255, 140): 'b1', (140, 170, 82): 'b2',
              (173, 223, 140): 'b3', (140, 255, 0): 'b4', (173, 186, 156): 'b5', (99, 186, 0): 'b6', (82, 154, 0): 'b7',
              (49, 101, 0): 'b8', (189, 223, 255): 'c0', (115, 207, 255): 'c1', (49, 85, 156): 'c2', (99, 154, 255): 'c3',
              (16, 117, 255): 'c4', (66, 117, 173): 'c5', (33, 69, 115): 'c6', (0, 32, 115): 'c7', (0, 16, 66): 'c8',
              (173, 255, 255): 'd0', (82, 255, 255): 'd1', (0, 138, 189): 'd2', (82, 186, 206): 'd3', (0, 207, 255): 'd4',
              (66, 154, 173): 'd5', (0, 101, 140): 'd6', (0, 69, 82): 'd7', (0, 32, 49): 'd8', (206, 255, 239): 'e0',
              (173, 239, 222): 'e1', (49, 207, 173): 'e2', (82, 239, 189): 'e3', (0, 255, 206): 'e4', (115, 170, 173): 'e5',
              (0, 170, 156): 'e6', (0, 138, 115): 'e7', (0, 69, 49): 'e8', (173, 255, 173): 'f0', (115, 255, 115): 'f1',
              (99, 223, 66): 'f2', (0, 255, 0): 'f3', (33, 223, 33): 'f4', (82, 186, 82): 'f5', (0, 186, 0): 'f6',
              (0, 138, 0): 'f7', (33, 69, 33): 'f8', (255, 255, 255): '0f', (236, 236, 236): '1f', (218, 218, 218): '2f',
              (200, 200, 200): '3f', (182, 182, 182): '4f', (163, 163, 163): '5f', (145, 145, 145): '6f', (127, 127, 127): '7f',
              (109, 109, 109): '8f', (91, 91, 91): '9f', (72, 72, 72): 'af', (54, 54, 54): 'bf', (36, 36, 36): 'cf',
              (18, 18, 18): 'df', (0, 0, 0): 'ef'}


colors_list = []
for keys in color_dict.keys():
    colors_list.append(list(keys))


# 输入一个颜色和色表，返回一个色表中最接近的颜色
def get_closest_color(color, colors_list):
    colors = np.array(colors_list)
    color = np.array(color)
    # 计算欧氏距离，找出原色值与色表中最接近的颜色
    distances = np.sqrt(np.sum((colors-color)**2, axis=1))
    index_of_nearest = np.where(distances == np.amin(distances))
    closest_color = colors[index_of_nearest][0]
    return closest_color


# 将正方形图片缩为32*32分辨率的jpg文件,并使用get_closest_color进行色值转换
# source_file,target_file = file path
def img_resize(filename):
    image = Image.open(filename)
    resized_image = np.array(image.resize((32, 32), resample=Image.LANCZOS), dtype=np.int)
    for i in range(32):
        for j in range(32):
            resized_image[i][j][0:3] = get_closest_color(resized_image[i][j][0:3], colors_list)
    Image.fromarray((np.array(resized_image)).astype('uint8')).save('temp_'+filename)
    return 'temp_'+filename


# 使用k-means算法压缩图像颜色到15种(不使用透明色)
# target_file = 32*32 img
def compress_colors(filename):
    image = Image.open(filename)
    data_origin = np.array(image)

    # 转为二维数组
    data = data_origin.reshape(-1, 3)

    # 使用k-means将所有颜色色值聚类为15种
    kmeans_predicter = KMeans(n_clusters=15)
    kmeans_predicter.fit(data)

    # 使用聚类后像素的中心值替换原像素颜色
    temp = kmeans_predicter.predict(data)
    data_new_temp = kmeans_predicter.cluster_centers_[temp]
    data_new = []

    palette_list = []

    # 聚类的中心值与原色表色值有误差，需要重新获取一次最接近颜色
    for i in data_new_temp:
        new_color = get_closest_color(list(i), colors_list)
        data_new.append(new_color)

    # 生成一个15种色值及对应标签的列表
    for i in data_new:
        color = tuple(list(i))
        if color not in palette_list:
            palette_list.append(color)
    for i in range(len(palette_list)):
        palette_list[i] = [list(palette_list[i]), color_dict[(palette_list[i])]]

    # 生成一个仅包含15种颜色的图片
    data_new = np.array(data_new)
    data_new.shape = data_origin.shape
    Image.fromarray((np.array(data_new, dtype=int)).astype('uint8')).save('temp_'+ filename)
    return palette_list


# 将png转换为jpg，放弃png中的A通道
def png_to_jpg(filename):
    image = Image.open(filename)
    image = image.convert('RGB')
    output_name = filename.split('.')[0] + '.jpg'
    image.save(output_name)
    return output_name


# 16进制转byte
def hex_to_byte(hexstr):
    byteList = []
    if len(hexstr) % 2 == 1:
        hexstr = hexstr + "0"
    for i in range(0, len(hexstr), 2):
        byte = int(hexstr[i: i + 2], 16)
        byteList.append(byte)
    return bytes(byteList)


def img_to_qr(filename):
    title = "6100" * 20
    author = "6100" * 9
    place = "6100" * 9
    prefix = title + "0000b6ec" + author + "000044c5" + place + "00001931"
    interfix = "cc0a090000"
    # 主色调颜色的字符串，取15种颜色的对应标签，长度为30位字符串
    palette = ''
    # 主色调颜色及标签列表 list of [color,labe]
    palette_list = compress_colors(img_resize(filename))
    # 单独抽离出主色调颜色
    palette_colors_list = []
    # 单独抽离出主色调标签
    label_list = []
    for color, label in palette_list:
        palette += label
        palette_colors_list.append(color)
        label_list.append(label)


    image = np.array(Image.open('temp_temp_' + filename), dtype=np.int)
    canvas = ''
    # 通过两次循环，将读取的所有像素点的色值转换为主色调列表中最接近的颜色，并给出最接近颜色在主色调列表里的16进制索引下标
    for i in range(0,32):
        for j in range(0, 16):
            # 读取的图片颜色用get_closet_color函数重新计算一次最接近颜色，色表用主色调列表
            color = get_closest_color(image[i][j * 2 + 1], palette_colors_list)
            label = palette_list[palette_colors_list.index(list(color))][1]
            # 给出color在主色调列表的索引下标，并转为16进制
            index = hex(label_list.index(label))[2:]
            canvas = canvas + hex(int(index, 16))[2:]
            color = get_closest_color(image[i][j * 2], palette_colors_list)
            label = palette_list[palette_colors_list.index(list(color))][1]
            index = hex(label_list.index(label))[2:]
            canvas = canvas + hex(int(index, 16))[2:]

    # 编码，将字符串转为bytelist
    byte_array = hex_to_byte(prefix + palette + interfix + canvas)
    qr_code = pyqrcode.QRCode(byte_array, error='M')
    qr_code.png('qrcode_' + filename)
    os.remove('temp_'+filename)
    os.remove('temp_temp_'+filename)



def main():
    files = os.listdir(os.getcwd())
    for i in files:
        if i.split('.')[-1] == 'jpg':
            img_to_qr(i)


if __name__ == '__main__':
    main()

















