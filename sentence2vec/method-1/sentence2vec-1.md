# Sentence2Vec
[TOC]

---
## 前言
&emsp;&emsp;本文是对论文[A Simple but Tough-to-Beat Baseline for Sentence Embeddings][1]中算法的简要描述，具体细节请参考[代码实现][2]。

## 算法介绍

&emsp;&emsp;1. **对一个句子中所有词的词向量进行加权平均**，每个词向量的权重可以表示为$\frac{a}{a+p(w)}$，其中$a$为参数，$p(w)$为词$w$的频率。
&emsp;&emsp;2. **使用PCA/SVD对向量值进行修改**。

&emsp;&emsp;*算法具体描述如下:*

<div align=center>
![这里写图片描述](http://img.blog.csdn.net/20180104200245855?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvV2Fsa2VyX0hhbw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


&emsp;&emsp;**算法细节**以及**代码实现**参考[github][2]。

---

## 参考文献
1. [A Simple but Tough-to-Beat Baseline for Sentence Embeddings][1]


[1]: https://openreview.net/pdf?id=SyK00v5xx
[2]: https://github.com/walkeao/NLP/tree/master/sentence2vec/method-1
