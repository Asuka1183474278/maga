import numpy as np
import LM
# 16个传感器位置
SensorPosition = np.array([
    [[0.12], [0.12], [0]],
    [0.12], [0.04], [0],
    [[0.12], [-0.04], [0]],
    [[0.12], [-0.12], [0]],
    [[0.04], [0.12], [0]],
    [[0.04], [0.04], [0]],
    [[0.04], [-0.04], [0]],
    [[0.04], [-0.12], [0]],
    [[-0.04], [0.12], [0]],
    [[-0.04], [0.04], [0]],
    [[-0.04], [-0.04], [0]],
    [[-0.04], [-0.12], [0]],
    [[-0.12], [0.12], [0]],
    [[-0.12], [0.04], [0]],
    [[-0.12], [-0.04], [0]],
    [[-0.12], [-0.12], [0]]
     ])

#%%

# 磁铁位置范围
length=200  #测试点个数
x11 = -0.12; x12 = 0.12;
y11 = -0.12; y12 = 0.12;
z11 = 0.05; z12 = 0.25;

xx1 = np.array([[x11], [y11], [z11], [0], [0]])
xx2 = np.array([[x12], [y12], [z12], [np.pi/2], [np.pi/2]])
Traj = np.empty((200, 5,1))
for i in range(length):
    Traj[i, :] = xx1 + (xx2 -xx1) * np.random.rand(5,1)

#%%

# 磁矩M
mag_diameter = 6.05e-3
mag_length = 1.25e-3
mag_BR = 14800
mag_BT = mag_BR* (mag_diameter**2) * (mag_length / 16)
M = np.empty((200,3,1))
for i in range(length):
    M[i, :] = np.array([np.sin(Traj[i, 3]) * np.cos(Traj[i, 4]), np.sin(Traj[i, 3]) * np.sin(Traj[i, 4]), np.cos(Traj[i, 4])])



#%%

theoryData = np.empty((16, 200, 3, 1))
for j in range(16):
    for i in range(200):
        R = np.linalg.norm(SensorPosition[j] - Traj[i, 0:3])
        theoryData[j, i] = (mag_BT * 3 * np.dot(M[i].T, SensorPosition[j] - Traj[i, 0:3])* (SensorPosition[j] - Traj[i, 0:3])- (R **2) * M[i] )/ (R ** 5)


#%%

# 16个传感器增益
Gain = np.empty((16, 3, 1),dtype=int)
for i in range(16):
    Gain[i] = np.array([[1800], [1800], [1800]]) +  200 * np.random.rand(3,1)

#传感器偏置
Offset = np.empty((16, 3, 1))
for i in range(16):
    Offset[i] = np.random.uniform(-0.05, 0.05, (3,1))

#传感器角度 Ψ Θ Φ
H = np.zeros((16,3,1))

for i in range(16):
    LM.main(Gain[i], H[i], Offset[i], theoryData[i])