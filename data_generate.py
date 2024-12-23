# import numpy as np
# import pickle
# from multiprocessing import Pool
# from itertools import product
#
# from n_body import detect_resonance
#
# # 定义参数组合
# param_v_earth = [0.5, 1.0, 1.5]
# param_v_hw = [0.0, 0.1, 0.2]
# param_v_angle = [0.0, np.pi/4, np.pi/2]
#
# param_combinations = list(product(param_v_earth, param_v_hw, param_v_angle))
#
# # 定义并行执行的函数
# def process_combination(v_e, v_h, v_a):
#     return detect_resonance(v_e, v_h, v_a)
#
# # 使用 multiprocessing.Pool 进行并行计算
# with Pool(processes=32) as pool:
#     results_bool = pool.starmap(process_combination, param_combinations)
#
# # 将结果转换为字典
# results_dict = {}
# for combo, res_val in zip(param_combinations, results_bool):
#     results_dict[combo] = res_val
#
# # 输出模拟结果
# for combo, val in results_dict.items():
#     print(f"参数: {combo}, 是否共振: {val}")
#
# # 保存为 pkl 文件
# with open("resonance_results.pkl", "wb") as f:
#     pickle.dump(results_dict, f)
import numpy as np
from n_body import detect_resonance

# 测试用的单一参数
v_earth_test = 1
v_hw_test = 1
v_angle_test = 0

# 调用函数并打印结果
res = detect_resonance(v_earth_test, v_hw_test, v_angle_test)
print(f"测试参数: (v_earth={v_earth_test}, v_hw={v_hw_test}, v_angle={v_angle_test:.2f}) -> 是否共振: {res}")