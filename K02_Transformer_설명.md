# **초보자를 위한 Transformer 이해하기**

---

## **1. Transformer란?**
Transformer는 **컴퓨터가 인간의 언어를 이해하고 생성하도록 돕는 AI 모델**입니다. **ChatGPT, 구글 번역, 음성 비서(Siri)** 등에 사용됩니다.

### **비유:**  
퍼즐을 맞출 때 **한 조각씩 놓는 대신**, Transformer는 **전체 퍼즐을 한 번에 보고 최적의 배치를 찾습니다.**  

이 덕분에 **더 빠르고 더 스마트하게** 작동합니다.

---

## **2. Transformer는 어떻게 작동할까?**
Transformer는 **문장을 한 번에 처리**합니다.  
Transformer는 두 가지 주요 구성 요소로 이루어져 있습니다:
1. **인코더(Encoder)** → 입력을 이해함 (문장을 읽는 것과 같음).  
2. **디코더(Decoder)** → 출력을 생성함 (문장을 번역하는 것과 같음).  

🔹 **예제:**  
다음 문장을 입력하면:  
➡️ *"나는 아이스크림을 좋아해. 왜냐하면 그것은 달콤하기 때문이야."*  
Transformer는 **"아이스크림"과 "달콤하다"**가 서로 연관되어 있음을 학습합니다.

---

## **3. Transformer의 단계별 동작 과정**
Transformer는 **세 가지 주요 과정**을 거칩니다:

### **1단계: 입력 임베딩 (단어를 숫자로 변환하기)**
컴퓨터는 단어를 이해하지 못하므로, **단어를 숫자로 변환**해야 합니다.  
예를 들면,  
- "나는" → **[0.2, 0.5, 0.1]**  
- "좋아해" → **[0.7, 0.3, 0.9]**  
- "아이스크림" → **[0.8, 0.4, 0.6]**  

각 단어는 **해당 의미를 포함한 벡터(숫자 리스트)**로 변환됩니다.

---

### **2단계: 위치 인코딩 (단어 순서 기억하기)**
Transformer는 **단어의 순서를 자연스럽게 읽지 않기 때문에**, 
**어떤 단어가 먼저 나왔는지 기억하는 정보(Positional Encoding)를 추가**합니다.  

예를 들면:
```
"나는 아이스크림을 좋아해."  →  [단어 벡터] + [위치 정보]
```
이 과정은 Transformer가 **문장의 구조를 이해하는 데 도움을 줍니다.**

---

### **3단계: 자기 주의(Self-Attention, 중요한 단어 찾기)**
💡 **책에서 중요한 문장을 형광펜으로 표시하는 것과 유사함.**  
Transformer는 문장을 이해할 때 **어떤 단어가 중요한지 스스로 결정**합니다.  

🔹 **예제:**  
- 문장: *"고양이가 매트 위에 앉아 있었고, 그것은 잠들었다."*  
- Transformer는 **"그것(it)"이 "고양이(cat)"을 의미한다는 것**을 자동으로 이해합니다.

Transformer는 **모든 단어를 서로 비교**하고 **각 단어의 중요도를 평가(Attention Score)** 합니다.

| 단어 1 | 단어 2 | 중요도 (Attention Score) |
|--------|--------|----------------|
| 고양이 | 그것  | 0.9 (강한 연관)    |
| 매트  | 그것  | 0.2 (약한 연관)    |

Transformer는 **"그것(it)"이 "고양이(cat)"와 관련이 높다는 것**을 인식하여 더 집중합니다.

---

### **4단계: 피드포워드 신경망 (최종 이해 과정)**
주의(Self-Attention) 적용 후, 정보는 **신경망(Neural Network)**을 통과하며 더욱 정제됩니다.

이 과정을 거쳐 Transformer는 **더 정확한 단어 예측**을 할 수 있습니다.

---

### **5단계: 디코더 (출력 생성)**
**디코더(Decoder)**는 인코더에서 처리된 정보를 바탕으로 최종 출력을 생성합니다.  
예제:
- **입력:** "나는 아이스크림을 좋아해."
- **출력:** "I love ice cream."

디코더는 **단어를 하나씩 생성하며, 전체 문맥을 유지**합니다.

---

## **4. Transformer 간단한 예제 코드 (Python)**
Transformer의 기본 개념을 코드로 구현해 보겠습니다.

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention

class SimpleTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(SimpleTransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])

    def call(self, inputs):
        attn_output = self.attention(inputs, inputs)
        out1 = self.norm1(inputs + attn_output)  # Add & Normalize
        ffn_output = self.ffn(out1)
        return self.norm2(out1 + ffn_output)  # Add & Normalize

# 입력 예제: 단어 5개, 각 단어는 10개의 특성 값을 가짐
sample_input = tf.random.uniform((1, 5, 10))
transformer = SimpleTransformerBlock(embed_dim=10, num_heads=2, ff_dim=20)
output = transformer(sample_input)

print("Transformer 출력 크기:", output.shape)
```

---

## **5. 핵심 정리**
✔ **Transformer는 모든 단어를 동시에 처리**합니다 (RNN처럼 한 단어씩 처리하지 않음).  
✔ **Self-Attention을 사용해 중요한 단어를 찾음**.  
✔ **ChatGPT, 번역기, 음성 AI에서 사용됨**.  
✔ **더 빠르고, 더 똑똑하게 언어를 이해함**.  

---

## **6. GitHub에 업로드하는 방법**
1. **이 파일을 `transformer_explanation.md`로 저장**합니다.
2. **GitHub에 접속하여 새 저장소(Repository)를 만듭니다.**
3. **"Add File" → "Upload File"을 클릭하고 `transformer_explanation.md`를 업로드합니다.**
4. **"Commit Changes" 버튼을 클릭하여 저장소에 반영합니다.**
5. 이제 GitHub에서 Markdown 형식으로 정리된 문서를 볼 수 있습니다!

---

💡 **이제 GitHub에 업로드할 준비가 완료되었습니다! 🚀**

