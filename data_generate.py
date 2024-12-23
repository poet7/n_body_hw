import numpy as np
import pickle
from multiprocessing import Pool
from itertools import product
from tqdm import tqdm
from n_body import detect_resonance

# 定义参数范围并生成参数组合
param_v_earth = np.linspace(0.5, 2.0, 4)  # 4个均匀分布值，例如 [0.5, 1.0, 1.5, 2.0]
param_v_hw = np.linspace(0.0, 0.5, 6)     # 6个均匀分布值，例如 [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
param_v_angle = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)  # 8个角度均匀分布

param_combinations = list(product(param_v_earth, param_v_hw, param_v_angle))

# 定义并行执行的函数
def process_combination(v_e, v_h, v_a):
    return detect_resonance(v_e, v_h, v_a)

# 保存结果为文件
def save_results(results_dict, filename="resonance_results.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(results_dict, f)
    print(f"结果已保存到文件: {filename}")

# 加载结果文件
def load_results(filename="resonance_results.pkl"):
    with open(filename, "rb") as f:
        results_dict = pickle.load(f)
    return results_dict

if __name__ == '__main__':
    # 使用 multiprocessing.Pool 进行并行计算
    with Pool(processes=32) as pool:
        # 使用 tqdm 包裹 starmap 函数，显示进度条
        results_bool = list(tqdm(pool.starmap(process_combination, param_combinations),
                                 total=len(param_combinations),
                                 desc="计算进度"))

    # 将结果转换为字典
    results_dict = {combo: res_val for combo, res_val in zip(param_combinations, results_bool)}

    # 输出模拟结果
    for combo, val in results_dict.items():
        print(f"参数: {combo}, 是否共振: {val}")

    # 保存结果
    save_results(results_dict)

    # 示例加载结果文件
    loaded_results = load_results()
    print(f"加载的结果样例: {list(loaded_results.items())[:5]}")  # 打印部分结果
