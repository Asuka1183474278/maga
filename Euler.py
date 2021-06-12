from math import cos, sin, pi, atan2 ,asin
import numpy as np
x, y, z  = 0 ,0 ,0 # 单位为角度
x, y, z  = x*pi/180 ,y*pi/180 ,z*pi/180
r11 ,r12 ,r13 = cos(z)*cos(y), cos(z)*sin(y)*sin(x) - sin(z)*cos(x), cos(z)*sin(y)*cos(x) + sin(z)*sin(x)
r21 ,r22 ,r23 = sin(z)*cos(y), sin(z)*sin(y)*sin(x) + cos(z)*cos(x), sin(z)*sin(y)*cos(x) - cos(z)*sin(x)
r31 ,r32 ,r33 = -sin(y), cos(y)*sin(x), cos(y)*cos(x)
R = np.array([[r11,r12,r13],
              [r21,r22,r23],
              [r31,r32,r33]])
print('欧拉角({0:f}, {1:f}, {2:f})转换为旋转矩阵'.format(x*180/pi, y*180/pi, z*180/pi))

print(R.shape)
