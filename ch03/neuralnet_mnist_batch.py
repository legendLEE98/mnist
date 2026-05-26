import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sigmoid import sigmoid
from softmax import softmax
from dataset.mnist import load_mnist
from PIL import Image

# 데이터 채우는 함수.
# 이걸 해야 x_train에 데이터가, t_train에 값이 생성됨
def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label= False)
    return x_test, t_test

# 미리 학습된 가중치 호출
def init_network():
    with open(os.path.dirname(__file__) + "/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network # network 값엔 딕셔너리 {'W1' : ~~~, 'W2': ~~~ , } 등이 들어있음

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2 ,b3 = network['b1'], network['b2'], network['b3']

    # np.dot = 행렬 곱 
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))