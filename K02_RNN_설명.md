# **순환 신경망(RNN) 이해하기**

---

## **1. RNN이란?**
**순환 신경망(Recurrent Neural Network, RNN)**은 **연속적인 데이터**를 처리하면서 이전 입력을 기억하는 인공지능 모델입니다.

### **비유:**  
책을 읽을 때, **이전 페이지를 기억하지 못하면 이야기의 흐름을 이해하기 어렵습니다.**  
RNN도 마찬가지로 **이전 입력을 기억하여 더 나은 예측을 수행합니다.**

일반적인 신경망은 입력을 **독립적으로** 처리하지만, RNN은 **히든 스테이트(hidden state)** 라는 개념을 사용하여 과거 정보를 계속 유지합니다.

---

## **2. 왜 RNN이 필요할까?**
### **기존 신경망의 한계:**
- **각 입력을 독립적으로 처리**하여 **이전 입력을 기억하지 못함**.
- **연속적인 데이터(예: 문장, 주식 가격, 음악 등) 처리에 취약함**.
- **메모리 기능이 없어서 시계열 데이터나 언어 기반 작업에 적합하지 않음**.

### **RNN이 해결하는 방법:**
- RNN은 **과거 정보를 히든 스테이트를 통해 유지**합니다.
- **텍스트, 음성, 시계열 예측** 등에서 뛰어난 성능을 보입니다.

---

## **3. RNN의 동작 원리**
RNN은 데이터를 **시간 순서대로 한 단계씩 처리**하며, **각 단계에서 히든 스테이트를 업데이트**합니다.

### **RNN 수학적 표현:**
각 타임스텝에서 RNN은 **히든 스테이트를 업데이트**합니다.

\[
h_t = f(W_x x_t + W_h h_{t-1} + b)
\]

여기서:
- \( h_t \) = **현재 히든 스테이트** (이전 입력 정보 포함)
- \( x_t \) = **현재 입력**
- \( h_{t-1} \) = **이전 히든 스테이트**
- \( W_x, W_h \) = **학습되는 가중치**
- \( b \) = **바이어스**
- \( f \) = **활성화 함수 (예: tanh, ReLU)**

히든 스테이트는 **과거 정보를 저장하고, 이를 다음 단계로 전달**합니다.

---

## **4. RNN 예제 코드 (Python)**
TensorFlow/Keras를 사용한 **간단한 RNN 모델**입니다.

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding

# 샘플 텍스트 데이터
text = "hello world"
chars = sorted(set(text))  # 고유 문자 리스트
char_to_idx = {c: i for i, c in enumerate(chars)}  # 문자 → 숫자 변환
idx_to_char = {i: c for i, c in enumerate(chars)}  # 숫자 → 문자 변환

# 텍스트 데이터를 숫자로 변환
sequence_length = 3
X, y = [], []

for i in range(len(text) - sequence_length):
    X.append([char_to_idx[c] for c in text[i:i + sequence_length]])
    y.append(char_to_idx[text[i + sequence_length]])

X = np.array(X)
y = np.array(y)

# RNN 모델 생성
model = Sequential([
    Embedding(input_dim=len(chars), output_dim=8, input_length=sequence_length),
    SimpleRNN(10, activation="relu"),
    Dense(len(chars), activation="softmax")
])

# 모델 학습
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, y, epochs=100, verbose=1)

# 다음 문자 예측
input_seq = np.array([[char_to_idx['h'], char_to_idx['e'], char_to_idx['l']]])  # "hel"
prediction = model.predict(input_seq)
predicted_char = idx_to_char[np.argmax(prediction)]

print(f"예측된 다음 문자: {predicted_char}")
```

---

## **5. RNN의 동작 과정 시각화**
### **🔁 순환 구조**
일반 신경망과 달리, RNN은 **이전 출력을 다음 입력에 반영**하여 문맥을 기억합니다.

```
타임스텝 1    타임스텝 2    타임스텝 3
    x1   →  h1  →  x2  →  h2  →  x3  →  h3
```
- **각 히든 스테이트(h)**가 **이전 정보를 저장**
- **최종 히든 스테이트**가 **예측에 활용됨**

---

## **6. 핵심 정리**
✔ **RNN은 히든 스테이트를 사용하여 과거 정보를 기억**합니다.  
✔ 각 타임스텝마다 **히든 스테이트를 업데이트**하여 문맥을 유지합니다.  
✔ 히든 스테이트 덕분에 RNN은 **텍스트 예측, 음성 인식, 음악 생성 등에 적합**합니다.  




