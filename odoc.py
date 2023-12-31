import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('./od_oc/predicted_img.png')
edge = cv2.Canny(img, 100, 200)

ans = []
for y in range(0, edge.shape[0]):
    for x in range(0, edge.shape[1]):
     if edge[y, x] != 0:
      ans = ans + [[x, y]]
ans = np.array(ans)

print(ans.shape)
print(ans)
#print(ans[0:10, :])

from PIL import Image
from pylab import imshow,save,show
from pylab import array
from pylab import plot
from pylab import title

# 一些点
x = ans[:,0]
y = ans[:,1]
import cv2
import numpy as np

# 通过OpenCV读取图片信息
img = cv2.imread('./od_oc/origin_img.png')

# 将制定像素点的数据设置为0, 要注意的是这三个参数对应的值是Blue, Green, Red。
#34,139,34
img[y, x] = [0, 0, 255]
img[y-1, x-1] = [0, 0, 255]
img[y-1, x+1] = [0, 0, 255]
img[y+1, x-1] = [0, 0, 255]
img[y+1, x+1] = [0, 0, 255]
# 将图像进行输出，使用show()也是可以显示的。
cv2.imshow("image", img)  # 显示图片，后面会讲解
cv2.waitKey(0)  # 等待按键
# cv2.imwrite('vis_odoc.png', img)
