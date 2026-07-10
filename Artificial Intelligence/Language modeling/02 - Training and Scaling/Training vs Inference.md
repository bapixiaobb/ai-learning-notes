#DeepLearning #NeuralNetwork #Transformer #LanguageModeling #Training #Inference

Training vs Inference 描述的是同一个模型在两个不同阶段的行为差异。

- **Training**：用数据和 loss 来学习参数。
- **Inference**：用已经训练好的参数，根据输入生成输出。

在 [[Language Modeling]] 中，这个区别尤其重要，因为 [[Decoder-Only Transformer]] 在训练时可以并行计算很多 positions 的 loss，但在生成时通常需要 autoregressively 一个 token 一个 token 地生成。

## 🧠 Core Idea

>[!note]
>Training 和 inference 使用的是同一个 [[Model Architecture]]，但目标不同。
>
>Training 的目标是更新参数：
>
>$$
>\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}
>$$
>
>Inference 的目标是使用固定参数进行预测或生成：
>
>$$
>p_\theta(x_{t+1} \mid x_{\leq t})
>$$

也就是说：

| Phase | Main Goal |
|---|---|
| Training | learn parameters |
| Inference | use learned parameters |

模型结构本身通常不变，但 forward pass 之后发生的事情不同。

## 🧩 High-Level Difference

| Aspect | Training | Inference |
|---|---|---|
| parameters | updated | fixed |
| input | training data | prompt / user input |
| output | logits + loss | logits + generated tokens |
| forward pass | yes | yes |
| backward pass | yes | no |
| optimizer step | yes | no |
| target labels | yes | no |
| dropout | usually on if used | off |
| KV cache | usually not the main mode | commonly used |
| goal | minimize loss | generate / predict |

>[!important]
>Inference 不是“模型开始工作”，training 也不是“没有 forward pass”。
>
>Training 和 inference 都会做 [[Forward Pass in Transformer]]；区别在于 training 还会计算 loss、backward、optimizer step，而 inference 不更新参数。

## 🏗️ Training

Training 是学习模型参数的过程。

给定 training data：

$$
x_1, x_2, \dots, x_T
$$

language model 做 forward pass，得到 logits：

$$
z_1, z_2, \dots, z_T
$$

然后用这些 logits 计算 loss，例如 [[Cross Entropy Loss]]：

$$
\mathcal{L}
=
-
\sum_t
\log p_\theta(x_{t+1} \mid x_{\leq t})
$$

接着通过 backward pass 计算 gradients：

$$
\nabla_\theta \mathcal{L}
$$

最后 optimizer 更新参数：

$$
\theta
\leftarrow
\theta
-
\eta \nabla_\theta \mathcal{L}
$$

>[!note]
>Training 的本质是：
>
>用 loss 衡量当前参数的预测错误，再通过 gradients 调整参数。

这里的 optimizer、learning rate、batch size、weight decay 等不属于 [[Model Architecture]]，而属于 [[Training Recipe]]。

## 🚀 Inference

Inference 是使用已经训练好的模型进行预测或生成。

在 inference 阶段，参数 $\theta$ 固定不变。

给定 prompt：

$$
x_1, x_2, \dots, x_T
$$

模型 forward pass 得到最后一个位置的 logits：

$$
z_T
$$

然后通过 [[Softmax]] 得到下一个 token 的概率分布：

$$
p_\theta(x_{T+1} \mid x_{\leq T})
=
\mathrm{softmax}(z_T)
$$

再根据某种 Decoding Strategy 选择下一个 token：

$$
x_{T+1}
$$

例如：

- greedy decoding；
- sampling；
- temperature sampling；
- top-k sampling；
- top-p sampling。

>[!note]
>Inference 的本质是：
>
>固定模型参数，用当前 context 计算 next-token distribution，然后选择或采样下一个 token。

## 🔁 Autoregressive Generation

在 [[Decoder-Only Transformer]] 中，inference 通常是 autoregressive 的 ([[Autoregressive Decoding]])。

也就是：

$$
x_1, \dots, x_T
\rightarrow
x_{T+1}
$$

然后把新 token 接回输入：

$$
x_1, \dots, x_T, x_{T+1}
\rightarrow
x_{T+2}
$$

继续重复：

$$
x_1, \dots, x_T, x_{T+1}, x_{T+2}
\rightarrow
x_{T+3}
$$

>[!note]
>每次生成新 token 都需要一次 forward pass。
>
>所以 inference 的延迟通常和生成 token 数量强相关。

这和训练不同：训练时可以一次性对一个 sequence 中的多个 positions 并行计算 loss；推理时生成 token 本身通常是 sequential 的。

## 📦 Training Data Flow

训练 decoder-only language model 时，输入和目标通常是同一个 token sequence 的 shifted version。

例如 sequence 是：

$$
[x_1, x_2, x_3, x_4, x_5]
$$

模型输入可以看成：

$$
[x_1, x_2, x_3, x_4]
$$

预测目标是：

$$
[x_2, x_3, x_4, x_5]
$$

也就是说：

| Position | Visible Prefix | Target |
|---|---|---|
| 1 | $x_1$ | $x_2$ |
| 2 | $x_1, x_2$ | $x_3$ |
| 3 | $x_1, x_2, x_3$ | $x_4$ |
| 4 | $x_1, x_2, x_3, x_4$ | $x_5$ |

>[!note]
>虽然每个 position 只能看见 prefix，但训练时可以用 [[Causal Mask]] 一次性并行计算所有 positions 的 logits 和 loss。
>
>这就是 Transformer 适合大规模训练的重要原因之一。

## 🧮 Inference Data Flow

推理时，模型通常只关心最后一个位置的 logits。

给定 prompt：

$$
[x_1, x_2, \dots, x_T]
$$

模型输出：

$$
z_T
$$

然后选择：

$$
x_{T+1}
$$

下一步输入变成：

$$
[x_1, x_2, \dots, x_T, x_{T+1}]
$$

再输出：

$$
z_{T+1}
$$

>[!note]
>训练时通常对所有 positions 计算 loss；推理时通常只使用最后位置的 logits 来生成下一个 token。
>
>这也是 training compute 和 inference compute 的结构差异之一。

## 🧵 Forward Pass in Both Phases

Training 和 inference 都有 forward pass。

### During Training

$$
\text{input tokens}
\rightarrow
\text{logits}
\rightarrow
\text{loss}
\rightarrow
\text{backward}
\rightarrow
\text{parameter update}
$$

### During Inference

$$
\text{prompt tokens}
\rightarrow
\text{logits}
\rightarrow
\text{decoding}
\rightarrow
\text{next token}
$$

>[!important]
>Forward pass 是两者共有的。
>
>Training 多了 loss、backward pass、optimizer step。
>
>Inference 多了 decoding strategy 和 autoregressive generation loop。

## 🔙 Backward Pass Only in Training

Backward pass 用来计算 loss 对参数的 gradients：

$$
\frac{\partial \mathcal{L}}{\partial \theta}
$$

这些 gradients 会被 optimizer 用来更新参数。

Inference 阶段通常不需要 backward pass，因为不训练参数。

>[!note]
>Inference 中也可以计算 gradients，例如做 adversarial analysis、interpretability 或 fine-tuning 相关操作。
>
>但普通文本生成 inference 不需要 backward pass。

## 🧠 Dropout and Mode Switching

一些 modules 在 training 和 inference 中行为不同，比如 [[Dropout]]。

Training 时，dropout 会随机置零一部分 activations：

$$
h_i \rightarrow 0
$$

Inference 时，dropout 通常关闭，模型使用完整 activations。

这也是为什么深度学习框架里通常有：

```python
model.train()
model.eval()
```
>[!note]  
`model.train()` 和 `model.eval()` 不会改变模型参数本身。
>
它们改变的是某些 layers 的行为，例如 dropout、batch norm 等。

在 modern LLM 中，dropout 是否使用取决于具体 training recipe，但 training / eval mode 的概念仍然重要。
## **🧊 KV Cache in Inference**

在 decoder-only Transformer 推理时，如果每生成一个新 token 都重新计算所有 previous tokens 的 $K,V$，会非常浪费。

[[KV Cache]] 会缓存之前 tokens 的 key 和 value：

$$  
K_{\leq t}, V_{\leq t}  
$$

生成下一个 token 时，只需要计算新 token 的 $Q,K,V$，然后让新 token attend to cached keys and values。

>[!note]  
KV cache 不改变模型的数学结果。
>
它改变的是 inference implementation，使 autoregressive generation 更高效。
>
KV cache 主要是 inference optimization，和 [[Systems for Language Models]] 关系很强。

## **⚖️ Parallelism Difference**

Training 和 inference 的 parallelism pattern 不同。

### **Training**

训练时，一个 sequence 中的所有 positions 可以并行处理：

$$  
[x_1, x_2, \dots, x_T]  
\rightarrow  
[z_1, z_2, \dots, z_T]  
$$

因为 causal mask 会保证每个位置不偷看 future tokens。

### **Inference**

推理时，生成 token 的过程通常不能完全并行：

$$  
x_{T+1}  
\rightarrow  
x_{T+2}  
\rightarrow  
x_{T+3}  
$$

因为后一个 token 依赖前一个已经生成出来的 token。

>[!note]  
这就是为什么 LLM inference latency 对用户体验很重要。
>
Training 更关注 throughput；inference 同时关注 latency、throughput 和 memory cost。

## **🧮 Compute and Memory Difference**

Training 通常比 inference 更 expensive，因为 training 需要：

- forward activations；
- loss computation；
- backward pass；
- gradients；
- optimizer states；
- parameter updates。

Inference 通常不需要：

- storing activations for backward；
- gradients；
- optimizer states。

但 inference 有自己的成本，例如：

- KV cache memory；
- autoregressive decoding latency；
- serving many concurrent requests；
- long-context attention cost。

>[!note]  
Training cost 和 inference cost 是 [[Systems for Language Models]] 中两条不同的分析线。
>
同一个 architecture，在 training 和 inference 中的 bottleneck 可能不同。

## **🎯 Example: Decoder-Only LM**

对于 [[Decoder-Only Transformer]]：

### **Training**

输入一整段 tokens：

$$  
[x_1, x_2, \dots, x_T]  
$$

模型并行输出：

$$  
[z_1, z_2, \dots, z_T]  
$$

计算：

$$  
p(x_2 \mid x_1),  
p(x_3 \mid x_{\leq 2}),  
\dots,  
p(x_T \mid x_{<T})  
$$

并对所有 positions 求 loss。

### **Inference**

输入 prompt：

$$  
[x_1, x_2, \dots, x_T]  
$$

模型输出最后位置 logits：

$$  
z_T  
$$

采样：

$$  
x_{T+1}  
$$

再继续生成。

>[!note]  
同一个 decoder-only architecture，在 training 时像是在“并行学习所有 next-token predictions”；在 inference 时像是在“逐步续写文本”。

## **🚫 Common Confusions**

### **1. Inference 不是没有 forward pass**

Inference 的核心就是 forward pass。  
只是不做 backward 和 optimizer step。

### **2. Training 不是一次只预测一个 token**

在 decoder-only LM 训练中，一个 sequence 的很多 positions 可以并行预测 next token。

### **3. Inference 通常只用最后一个 logits**

训练时用很多 positions 的 logits 计算 loss；推理生成下一个 token 时通常只用最后位置的 logits。

### **4. KV cache 不是训练目标**

KV cache 是推理加速机制，不是 architecture objective，也不是 loss function。

### **5. Decoding strategy 不等于 model architecture**

Temperature、top-k、top-p 改变的是如何从 logits 选择 token，不改变模型本身结构。

---

>[!summary]  My Understanding
[[Training vs Inference]] 的关键区别在于：training 用数据和 loss 更新参数，inference 使用固定参数生成输出。
>
两者都会做 [[Forward Pass in Transformer]]。
>
Training 还包括 loss computation、backward pass 和 optimizer step；inference 则包括 decoding strategy 和 autoregressive generation loop。
>
在 decoder-only language model 中，训练时可以并行计算多个 positions 的 next-token loss，而推理时通常需要一个 token 一个 token 地生成，因此 [[KV Cache]] 对 inference efficiency 非常重要。

## **🔗 Connections**

- [[Forward Pass in Transformer]]
- [[Decoder-Only Transformer]]
- [[Transformer Block]]
- [[Residual Stream]]
- [[Training Recipe]]
- [[Systems for Language Models]]
- [[Cross Entropy Loss]]
- [[Dropout]]
- [[KV Cache]]
- [[Resource Accounting]]