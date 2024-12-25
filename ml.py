import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# 使用原生 Keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam

class TimeoutException(Exception):
    """自定义超时异常"""

def load_and_filter_data(filename="resonance_results_101083.pkl"):
    """从文件载入数据，删除结果为 None 的记录"""
    with open(filename, "rb") as f:
        results_dict = pickle.load(f)
    # 过滤掉结果为 None 的记录
    filtered_data = {k: v for k, v in results_dict.items() if v is not None}
    return filtered_data

def prepare_dataset(results_dict):
    """将 (v_e, v_h, v_a, t_s) 转化为 X，(共振True/False) 转化为 y"""
    X, y = [], []
    for params, resonant in results_dict.items():
        # params 是 (v_earth, v_hw, v_angle, t_stop)
        # resonant 为布尔 True/False
        X.append(params)
        y.append(1 if resonant else 0)
    return np.array(X, dtype=float), np.array(y, dtype=int)

def build_model(input_dim):
    """创建简单的 MLP 分类模型"""
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == '__main__':
    # 1. 载入并过滤掉结果为 None 的记录
    data_dict = load_and_filter_data("resonance_results_101083.pkl")

    # 2. 转化为可训练的数组
    X, y = prepare_dataset(data_dict)
    print("有效数据量:", len(X))

    # 3. 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. 构建并训练模型
    model = build_model(input_dim=4)
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # 5. 简要评估
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n测试集准确率: {test_acc:.3f}")

    # 6. 保存模型
    model.save("resonance_model.h5")
    print("模型已保存为 resonance_model.h5")

    # 7. 演示加载模型并进行预测
    loaded_model = load_model("resonance_model.h5")
    sample_predictions = loaded_model.predict(X_test[:5])
    print("\n对测试集中前5条数据进行预测：")
    print("预测结果（sigmoid输出）:\n", sample_predictions.reshape(-1))
    print("实际标签:\n", y_test[:5])