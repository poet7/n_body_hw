import multiprocessing
import numpy as np
from scipy.integrate import solve_ivp
import time

class TimeoutException(Exception):
    pass

# 原先的分析函数
def _analyze_solution(sol):
    x_e, y_e = sol.y[4], sol.y[5]
    a_e = np.sqrt(x_e**2 + y_e**2)
    T_j = 1.0
    T_e = a_e**1.5
    period_ratio = T_e / T_j

    resonances = [2/1, 3/2, 5/3, 7/5]
    tolerance = 0.03

    for res in resonances:
        indices = np.where(np.abs(period_ratio - res) < tolerance)[0]
        if len(indices) > 10:
            nearby = period_ratio[indices]
            diff = np.diff(nearby)
            is_increasing = np.all(diff >= 0)
            is_decreasing = np.all(diff <= 0)
            if not (is_increasing or is_decreasing):
                return True
    return False

# 原先的核心计算函数
def _compute_resonance(v_earth, v_hw, v_angle, t_stop):
    G = 4 * np.pi**2
    M_sun = 1.0
    a_jupiter = 1.0
    v_jupiter = np.sqrt(G * M_sun / a_jupiter)
    a_earth = 2.0
    M_jupiter = 1e-3
    v_earth_stable = np.sqrt(G * M_sun / a_earth)

    vx_e = v_earth * np.cos(v_angle)
    vy_e = v_earth_stable + v_earth * np.sin(v_angle)
    y0 = [a_jupiter, 0.0, 0.0, v_jupiter, a_earth, 0.0, vx_e, vy_e]

    def drag_force(v, v_gas):
        return -(v - v_gas) / t_stop

    def equations(t, y):
        x_j, y_j, vx_j, vy_j, x_e, y_e, vx_e, vy_e = y
        dx = x_e - x_j
        dy = y_e - y_j
        r_je = np.sqrt(dx**2 + dy**2)
        r_j = np.sqrt(x_j**2 + y_j**2)
        r_e = np.sqrt(x_e**2 + y_e**2)

        ax_j = -G * M_sun * x_j / r_j**3
        ay_j = -G * M_sun * y_j / r_j**3
        ax_e = -G * M_sun * x_e / r_e**3 - G * M_jupiter * dx / r_je**3
        ay_e = -G * M_sun * y_e / r_e**3 - G * M_jupiter * dy / r_je**3

        v_k = np.sqrt(G * M_sun / r_e)
        v_gas = v_k - v_hw
        v_e = np.array([vx_e, vy_e])
        e_phi = np.array([-y_e, x_e]) / r_e
        e_phi /= np.linalg.norm(e_phi)
        a_drag = drag_force(v_e, v_gas * e_phi)

        ax_e += a_drag[0]
        ay_e += a_drag[1]
        return [vx_j, vy_j, ax_j, ay_j, vx_e, vy_e, ax_e, ay_e]

    t_span = (0, 25000)
    t_eval = np.linspace(*t_span, 500)
    sol = solve_ivp(equations, t_span, y0, t_eval=t_eval, rtol=1e-7, atol=1e-7, max_step=10)
    return _analyze_solution(sol)

# 用单独进程来执行实际计算
def _process_target(queue, v_earth, v_hw, v_angle, t_stop):
    try:
        res = _compute_resonance(v_earth, v_hw, v_angle, t_stop)
        queue.put(res)
    except Exception as e:
        queue.put(e)

def detect_resonance(v_earth, v_hw, v_angle, t_stop, timeout=240):
    """
    使用进程强制超时终止
    """
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_process_target, args=(q, v_earth, v_hw, v_angle, t_stop))
    p.start()

    start_time = time.time()
    while True:
        if not p.is_alive():
            # 子进程已退出
            break
        if time.time() - start_time > timeout:
            # 超时，强制杀死进程
            p.terminate()
            p.join()
            raise TimeoutException(f"detect_resonance timed out after {timeout} seconds")
        time.sleep(0.1)

    p.join()
    # 读取子进程结果
    result = q.get()
    if isinstance(result, Exception):
        # 若子进程中抛出异常，则在主进程中重新抛出
        raise result
    return result

if __name__ == '__main__':
    # 测试调用示例
    try:
        ans = detect_resonance(v_earth=1.0, v_hw=0.1, v_angle=np.pi/4, t_stop=100, timeout=5)
        print("共振结果:", ans)
    except TimeoutException as e:
        print("超时:", e)