import numpy as np
import logging
import timeout_decorator
from scipy.integrate import solve_ivp

class TimeoutException(Exception):
    """自定义超时异常"""

@timeout_decorator.timeout(480, timeout_exception=TimeoutException)  # 默认超时时间480秒，可自行修改
def detect_resonance(v_earth, v_hw, v_angle, t_stop):
    """
    使用timeout_decorator来限制运行时间，若超时则抛出TimeoutException
    """
    try:
        result = _compute_resonance(v_earth, v_hw, v_angle, t_stop)
    except TimeoutException:
        logging.error("计算超时")
        return None  # 超时返回默认值
    except Exception as e:
        logging.error(f"计算错误: {e}")
        return None
    return result


def _compute_resonance(v_earth, v_hw, v_angle, t_stop):
    """
    实际执行积分计算的函数
    """
    # 固定参数
    a_earth = 2.0
    M_jupiter = 1e-3

    # 常量定义
    G = 4 * np.pi**2
    M_sun = 1.0
    a_jupiter = 1.0
    v_jupiter = np.sqrt(G * M_sun / a_jupiter)
    v_earth_stable = np.sqrt(G * M_sun / a_earth)

    # 初始状态向量
    vx_e = v_earth * np.cos(v_angle)
    vy_e = v_earth_stable + v_earth * np.sin(v_angle)
    y0 = [
        a_jupiter, 0.0, 0.0, v_jupiter,
        a_earth,   0.0, vx_e, vy_e
    ]

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

    # 积分配置
    t_span = (0, 35000)
    t_eval = np.linspace(*t_span, 500)
    sol = solve_ivp(equations, t_span, y0, t_eval=t_eval, rtol=1e-7, atol=1e-7, max_step=10)

    result = _analyze_solution(sol)
    print(f"v_earth={v_earth}, v_hw={v_hw}, v_angle={v_angle}, t_stop={t_stop}, result={result}")
    return result

def _analyze_solution(sol):
    """
    提取解并检查是否存在共振
    """
    x_e, y_e = sol.y[4], sol.y[5]
    a_e = np.sqrt(x_e**2 + y_e**2)
    T_j = 1.0
    T_e = a_e**1.5
    period_ratio = T_e / T_j

    return _check_resonance(period_ratio, sol)

def _check_resonance(period_ratio, sol):
    """
    检查最后 1000 年内的周期比是否满足共振条件
    """
    resonances = [2/1, 3/2, 5/3, 7/5]
    tolerance = 0.25
    tolerance = 0.25
    final_time_window = 1000  # 最后1000年窗口

    # 找到最后 1000 年的时间范围
    final_indices = np.where(sol.t >= sol.t[-1] - final_time_window)[0]
    final_period_ratios = period_ratio[final_indices]

    for res in resonances:
        # 检查最后 1000 年内是否全部在容差范围内
        if np.all(np.abs(final_period_ratios - res) < tolerance):
            return True
    return False
