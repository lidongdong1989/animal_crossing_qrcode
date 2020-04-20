## 动物之森二维码生成器
实现任意图片转为动森支持的二维码，可通过NitendoSwitchOnline APP扫描二维码，将图片同步到动森我的设计中。

### 动森二维码限制
- 宽高均为32像素，共1024个像素点。
- 最多支持160种颜色。
- 每张画布最多允许使用16种颜色，包括1个透明色。

### 使用方法
将原始图放入脚本目录下，原图最好处理提前处理为正方形。
运行脚本，脚本目录下会生成一张qrcode_原文件名的图片。



### Background information
animal crossing canvas size: 32px*32px, 160 colors supported, including 144 colors, 15 grayscale, 1 transparent
Only up to 16 colors are allowed for a single canvas, including 1 transparent color

### How to use
- Place a JPG image in the script directory
- run script. python3 animal_crossing_qrcode.py
- The file name of the output is qrcode_filename.png
- open NitentoSwitchOnline，scan Qrcode


### todo
~~- support PNG~~
- auto cut
- support other kind of design,like clothes


