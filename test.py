import numpy as np
import matplotlib.pyplot as plt




tao = 10 ** -3
threshold_stop = 10 ** -15
threshold_step = 10 ** -15
threshold_residual = 10 ** -15
residual_memory = []


# construct a user function 理论值
def my_Func(params, input_data):
    G1 = params[0, 0]
    G2 = params[1, 0]
    G3 = params[2, 0]
    H1 = params[3, 0]
    H2 = params[4, 0]
    H3 = params[5, 0]
    O1 = params[6, 0]
    O2 = params[7, 0]
    O3 = params[8, 0]
    X = np.zeros((200, 3, 1))
    for i in range(0, 200):
        X[i] = np.diag(np.array([G1, G2, G3] * np.array([H1, H2, H3]))).dot(input_data[i])+ np.array([[O1], [O2], [O3]])
    return X


# generating the input_data and output_data,whose shape both is (num_data,3) 加高斯噪声
def generate_data(params, input_data):
    mid, sigma = 0, 0.5
    y = my_Func(params, input_data) + np.random.normal(mid, sigma, input_data.shape)
    return y


# calculating the derive of pointed parameter,whose shape is (num_data,3) 求导
def cal_deriv(params, input_data, param_index):
    params1 = params.copy()
    params2 = params.copy()
    params1[param_index, 0] += 0.000001
    params2[param_index, 0] -= 0.000001
    data_est_output1 = my_Func(params1, input_data)
    data_est_output2 = my_Func(params2, input_data)
    return (data_est_output1 - data_est_output2) / 0.000002


# calculating jacobian matrix,whose shape is (num_data,num_params，3) 雅克比
def cal_Jacobian(params, input_data):
    num_params = np.shape(params)[0]
    num_data = np.shape(input_data)[0]
    J = np.zeros((num_data, num_params, 3, 1))
    for i in range(0, 9):
        J[:, i] = np.array(cal_deriv(params, input_data, i))
    return J


# calculating residual, whose shape is (num_data,3)
def cal_residual(params, input_data, output_data):
    data_est_output = my_Func(params, input_data)
    residual = output_data - data_est_output
    return residual


'''    
#calculating Hessian matrix, whose shape is (num_params,num_params)
def cal_Hessian_LM(Jacobian,u,num_params):
    H = Jacobian.T.dot(Jacobian) + u*np.eye(num_params)
    return H

#calculating g, whose shape is (num_params,1)
def cal_g(Jacobian,residual):
    g = Jacobian.T.dot(residual)
    return g

#calculating s,whose shape is (num_params,1)
def cal_step(Hessian_LM,g):
    s = Hessian_LM.I.dot(g)
    return s

'''


# get the init u, using equation u=tao*max(Aii)
def get_init_u(A, tao):
    m = np.shape(A)[0]
    Aii = []
    for i in range(0, m):
        Aii.append(A[i, i])
    u = tao * max(Aii)
    return u


# LM algorithm
def LM(num_iter, params, input_data, output_data):
    num_params = np.shape(params)[0]  # the number of params
    k = 0  # set the init iter count is 0
    # calculating the init residual
    residual = cal_residual(params, input_data, output_data)
    # calculating the init Jocobian matrix
    Jacobian = cal_Jacobian(params, input_data).reshape(200,9,3)
    A0 = Jacobian[:, :, 0].T.dot(Jacobian[:, :, 0])# calculating the init A
    A1 = Jacobian[:, :, 1].T.dot(Jacobian[:, :, 1])
    A2 = Jacobian[:, :, 2].T.dot(Jacobian[:, :, 2])
    A = A0 + A1 + A2
    g0 = Jacobian[:, :, 0].T.dot(residual[:, 0])  # calculating the init gradient g
    g1 = Jacobian[:, :, 1].T.dot(residual[:, 1])
    g2 = Jacobian[:, :, 2].T.dot(residual[:, 2])
    g = g0 + g1 + g2

    stop = (np.linalg.norm(g, ord=np.inf) <= threshold_stop)  # set the init stop
    print(stop)
    u = get_init_u(A, tao)  # set the init u
    v = 2  # set the init v=2

    while ((not stop) and (k < num_iter)):
        k += 1
        while (1):
            Hessian_LM = A + u * np.eye(num_params)  # calculating Hessian matrix in LM
            step = np.linalg.inv(Hessian_LM).dot(g)  # calculating the update step
            if (np.linalg.norm(step) <= threshold_step):
                stop = True
            else:
                new_params = params + step  # update params using step
                new_residual = cal_residual(new_params, input_data, output_data)  # get new residual using new params
                rou = (np.linalg.norm(residual) ** 2 - np.linalg.norm(new_residual) ** 2) / (step.T.dot(u * step + g))
                if rou > 0:
                    params = new_params
                    residual = new_residual
                    residual_memory.append(np.linalg.norm(residual) ** 2)
                    # print (np.linalg.norm(new_residual)**2)
                    Jacobian = cal_Jacobian(params, input_data).reshape(200,9,3)  # recalculating Jacobian matrix with new params
                    A0 = Jacobian[:, :, 0].T.dot(Jacobian[:, :, 0])  # calculating the init A
                    A1 = Jacobian[:, :, 1].T.dot(Jacobian[:, :, 1])
                    A2 = Jacobian[:, :, 2].T.dot(Jacobian[:, :, 2])
                    A = A0 + A1 + A2
                    g0 = Jacobian[:, :, 0].T.dot(residual[:, 0])  # calculating the init gradient g
                    g1 = Jacobian[:, :, 1].T.dot(residual[:, 1])
                    g2 = Jacobian[:, :, 2].T.dot(residual[:, 2])
                    g = g0 + g1 + g2
                    stop = (np.linalg.norm(g, ord=np.inf) <= threshold_stop) or (
                                np.linalg.norm(residual) ** 2 <= threshold_residual)
                    u = u * max(1 / 3, 1 - (2 * rou - 1) ** 3)
                    v = 2
                else:
                    u = u * v
                    v = 2 * v
            if (rou > 0 or stop):
                break

    return params


def main(Gain, H, O,theoryData):
    # set the true params for generate_data() function
    params = np.zeros((9, 1))
    params[0, 0] = Gain[0,0]
    params[1, 0] = Gain[1,0]
    params[2, 0] = Gain[2,0]
    params[3, 0] = H[0,0]
    params[4, 0] = H[1,0]
    params[5, 0] = H[2,0]
    params[6, 0] = O[0,0]
    params[7, 0] = O[1,0]
    params[8, 0] = O[2,0]
    data_input = theoryData
    data_output = generate_data(params, data_input)
    # set the init params for LM algorithm
    params = np.zeros((9, 1))
    params[0, 0] = 200
    params[1, 0] = 200
    params[2, 0] = 200
    params[3, 0] = 0.93
    params[4, 0] = 0.94
    params[5, 0] = 0.95
    params[6, 0] = 0.01
    params[7, 0] = -0.005
    params[8, 0] = 0.0006
    # using LM algorithm estimate params
    num_iter = 200  # the number of iteration
    est_params = LM(num_iter, params, data_input, data_output)
    print(est_params)
