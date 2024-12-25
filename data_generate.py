import numpy as np
import pickle
from itertools import product
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from n_body import detect_resonance


# 定义并行执行的函数
def process_combination(params):
    v_e, v_h, v_a, t_s = params
    try:
        # 调用 detect_resonance 进行计算
        return params, detect_resonance(v_e, v_h, v_a, t_s)
    except Exception as e:
        print(f"参数组合 {params} 计算出错: {e}")
        return params, None


# 定义参数范围并生成参数组合
param_v_earth = np.linspace(0.0, 2.0, 5)  # 4个均匀分布值，例如 [0.5, 1.0, 1.5, 2.0]
param_v_hw = np.linspace(0.0, 2, 5)     # 4个均匀分布值，例如 [0.0, 0.1666, 0.3333, 0.5]
param_v_angle = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)  # 8个角度均匀分布
param_t_stop = np.linspace(50, 150, 3)     # 阻力时间尺度 [50, 100, 150]

param_combinations = list(product(param_v_earth, param_v_hw, param_v_angle, param_t_stop))


# 保存结果为文件
def save_results(results_dict, filename="resonance_results_5583.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(results_dict, f)
    print(f"结果已保存到文件: {filename}")


# 加载结果文件
def load_results(filename="resonance_results_5583"
                          ".pkl"):
    with open(filename, "rb") as f:
        results_dict = pickle.load(f)
    return results_dict


# 主程序入口
if __name__ == '__main__':
    # 进程数根据性能调整，设置为16以充分利用16核心
    num_workers = 16
    results_dict = {}

    # 创建进程池并分发任务
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_params = {executor.submit(process_combination, params): params for params in param_combinations}

        # 显示进度条
        with tqdm(total=len(param_combinations), desc="计算进度", dynamic_ncols=True) as pbar:
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    # 获取计算结果
                    params, res = future.result()
                    if res is not None:
                        results_dict[params] = res
                except Exception as e:
                    print(f"参数组合 {params} 计算出错: {e}")
                pbar.update(1)  # 更新进度条

    # 输出部分模拟结果
    print("\n计算完成，部分结果示例：")
    for combo, val in list(results_dict.items())[:5]:
        print(f"参数: {combo}, 是否共振: {val}")

    # 保存结果
    save_results(results_dict)

    # 示例加载结果文件
    loaded_results = load_results()
    print("\n加载结果文件的前5项：")
    print(list(loaded_results.items())[:5])
