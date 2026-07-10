#Tokenization #LanguageModeling

**定义**：BPE 是一种 subword tokenization 方法，从 bytes 或 characters 开始，反复合并高频相邻 token pair。

起点是把所有文本 -> UTF-8 bytes
## 例子

```text
t h e
-> th e
-> the
```
## **核心思想**

BPE 在 vocabulary size 和 sequence length 之间做 trade-off：

|**方法**|**vocabulary size**|**sequence length**|
|---|---|---|
|byte token|小|长|
|word token|大|短|
|BPE token|中等|中等|
