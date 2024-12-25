import numpy as np
import pickle
import matplotlib.pyplot as plt

def plot_results(filename):
    # 载入结果
    with open(filename, 'rb') as f:
        results_dict = pickle.load(f)

    # 分别存储三种结果的坐标
    x_true, y_true = [], []
    x_false, y_false = [], []
    x_none, y_none = [], []

    # 遍历每条数据
    for params, res in results_dict.items():
        v_e, v_h, v_a, t_s = params  # 分别对应 v_earth, v_hw, v_angle, t_stop
        # 假设只绘制 (v_earth, v_hw) -> x,y
        if res is True:
            x_true.append(v_e)
            y_true.append(v_h)
        elif res is False:
            x_false.append(v_e)
            y_false.append(v_h)
        else:  # None 或其他
            x_none.append(v_e)
            y_none.append(v_h)

    # 开始绘图
    plt.figure(figsize=(6, 5))
    # True: 用圆圈
    plt.scatter(x_true, y_true, marker='o', color='green', label='True')
    # False: 用正方形
    plt.scatter(x_false, y_false, marker='s', color='red', label='False')
    # None: 用叉号
    plt.scatter(x_none, y_none, marker='x', color='blue', label='None')

    plt.xlabel('v_earth')
    plt.ylabel('v_hw')
    plt.title('共振结果示意 (True / False / None)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    plot_results("resonance_results_101083.pkl")