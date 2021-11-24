# Project 2: Word2vec

## 1 Introduction

​	Word2vec model learns continuous vector for each words, and it has been shown that compared with one-hot vector, much more semantic meanings are carried. Moreover, word2vec is the necessary foundation of various down-stream NLP tasks (e.g. machine translation). Generally, there are two kinds of word2vec models, including Continuous Bag-Of-Word (CBOW) model and Skip-Gram (SG) model.

​	In this project, the CBOW model is implemented, which is a simple way to obtain the vector representations of words by maximizing the probability of center word given the context words. The structure of CBOW model is shown in the following figure:

​	<img src="D:\OneDrive - sjtu.edu.cn\大三上\自然语言处理\Projects&Assignments\Project2\fig\CBOW.png" style="zoom:60%;" />

## 2 Derivation of CBOW Model

### 2.1 Symbol Definition

​	The meanings of symbols used in the derivation are shown in the following table:

|  Symbol  |                Meaning                 |  Shape   |
| :------: | :------------------------------------: | :------: |
|   $V$    |            Vocabulary size             |    /     |
|   $N$    |           Hidden layer size            |    /     |
|   $C$    |              Context size              |    /     |
| $\delta$ |             Learning rate              |    /     |
|   $L$    |                  loss                  |    /     |
|  $W_1$   | Weight between input and hidden layer  | $(V, N)$ |
|  $W_2$   | Weight between hidden and output layer | $(N, V)$ |
|   $x$    |    One-hot vectors of context words    | $(V,C)$  |
|   $t$    |     One-hot vector of center word      | $(V,1)$  |
|   $h$    |              Hidden layer              | $(N,1)$  |
|   $u$    |          Score for each word           | $(V,1)$  |
|   $y$    |           Output probability           | $(V,1)$  |

### 2.2 Forward Progress

#### 2.2.1 Derivation

​	Firstly, the hidden layer is computed by taking the average of the product of $W_1$ and $x$:
$$
h=\frac{1}{C}W_1^T(x_1+x_2+\cdots+x_C)
$$
​	Secondly, the score of each word can be obtained by simply multiply $W_2$ and $h$:
$$
u=W_2^Th
$$

#### 2.2.2 Implementation

​	The forward progress can be implemented by the following code:

```python
h = np.dot(self.W1.T, x)  # N x C
h = np.mean(h, axis=1).reshape(-1, 1)  # N x 1
u = np.dot(self.W2.T, h)  # V x 1
```

### 2.3 Loss Calculation

#### 2.3.1 Derivation

​	Firstly, in order to compute the loss, we should compute the probability from the scores using the softmax function:
$$
y_i=\frac{\text{exp}(u_i)}{\sum_{j=0}^V \text{exp}(u_j)}
$$
​	Secondly, the cross-entropy loss is used, thus can be computed by :
$$
L=-\text{log}(y_{i^*})=-u_{i^*}+\text{log}\sum_{j=0}^V \text{exp}(u_j)
$$

#### 2.3.2 Implementation

​	The computation of loss can be implemented by the following code:

```python
y = softmax(u)
loss = -(
    np.sum(u*t) -
    np.log(np.sum(np.exp(u)))
)
```

### 2.4 Backward Progress

#### 2.4.1 Derivation

​	In order to minimize the loss and learn the word-vectors, $W_1$ and $W_2$ should be updated in each step using back-propagation. 

​	Take the derivative of L with regard to u, we obtain:
$$
\frac{\partial L}{\partial u}=y-t=e
$$
​	Next, the derivative of L with regard to $W_2$ can be obtained by:
$$
\frac{\partial L}{\partial W_2}=\frac{\partial L}{\partial u}\times \frac{\partial u}{\partial W_2}=he^T
$$
​	Similarly, the derivative of L with regard to $W_1$ can be obtained by:
$$
\frac{\partial L}{\partial W_1}=\frac{\partial L}{\partial u}\times \frac{\partial u}{\partial h}\times \frac{\partial h}{\partial W_1}=\frac{1}{C}\left( x_1+x_2+\cdots +x_C \right) \left( W_2e \right) ^T
$$
​	Therefore, $W_1$ and $W_2$ can be updated by:
$$
W_1=W_1-\delta\frac{\partial L}{\partial W_1}=W_1-\delta\frac{1}{C}\left( x_1+x_2+\cdots +x_C \right) \left( W_2e \right) ^T
\\
W_2=W_2-\delta \frac{\partial L}{\partial W_2}=W_2-\delta he^T
$$

#### 2.4.2 Implementation

​	The backward updating progress can be implemented by the following code:

```python
e = np.subtract(y, t)  # V x 1
self.W1 -= np.dot(x.sum(axis=1).reshape(-1, 1),
                  (1 / x.shape[1] * learning_rate) *
                  np.dot(self.W2, e).T)
self.W2 -= learning_rate * np.dot(h, e.T)
```

## 3 Results

### 3.1 Test Results

#### 3.1.1 Test 1

Test 1 is a sanity check using very small amount of data.

```
Token number: 50
Vocab size: 21
Epoch 1, loss: 2.96. Cost 0.0 min
Epoch 2, loss: 1.99. Cost 0.0 min
Epoch 3, loss: 1.46. Cost 0.0 min
Epoch 4, loss: 1.16. Cost 0.0 min
Epoch 5, loss: 0.94. Cost 0.0 min
Epoch 6, loss: 0.82. Cost 0.0 min
Epoch 7, loss: 0.74. Cost 0.0 min
Epoch 8, loss: 0.70. Cost 0.0 min
Epoch 9, loss: 0.82. Cost 0.0 min
Epoch 10, loss: 1.09. Cost 0.0 min
[('i', 1.0), ('he', 0.9871080376153758), ('she', 0.7916710150878632), ('read', 0.5932057410388843), ('to', 0.561705216393376)]
[('he', 1.0), ('i', 0.9871080376153758), ('she', 0.7600563117313381), ('to', 0.6236959230648946), ('read', 0.5664093733127676)]
[('she', 1.0), ('now', 0.9166363811901121), ('i', 0.7916710150878632), ('he', 0.7600563117313381), ('will', 0.7217754180931892)]
```

The final loss is $1.09$, and the similarity between 'i', 'he', 'she' is high.

#### 3.1.2 Test 2

​	In Test 2, the CBOW model is trained on large dataset for much longer time, and the final result is shown in the following figure:

![](D:\OneDrive - sjtu.edu.cn\大三上\自然语言处理\Projects&Assignments\Project2\fig\Test2.png)

The final loss is $6.92$, which is lower than $7.0$, meaning that the model is well-trained.

#### 3.1.3 Test3

​	In Test3, the spearman correlation and pearson correlation are computed to evaluate whether the similar words have similar vectors:

![](D:\OneDrive - sjtu.edu.cn\大三上\自然语言处理\Projects&Assignments\Project2\fig\Test3.png)

The spearman correlation is $0.397$ while the pearson correlation is $0.551$, larger than $0.3$ and $0.4$, respectively.

### 3.2 Visualization

​	Inspired by the assignment 2 of [Stanford CS224N: NLP with Deep Learning](https://www.youtube.com/watch?v=8rXD5-xhemo&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z), the word vectors trained by our CBOW model are visualized in the following figure:

<img src="D:\OneDrive - sjtu.edu.cn\大三上\自然语言处理\Projects&Assignments\Project2\fig\word_vectors_2.png" style="zoom:70%;" />

​	I choose several pairs of words that have similar meanings, and reduce the dimension of each vector to $2$ by using Singular Value Decomposition, making it possible to put them on the 2-D surface. It is really interesting to see that the similar words tend to stay close in the figure (e.g. film & movie, camera & photography, kid & child, ...)



**I guarantee that everything of the code and report is done by myself without using any complex open-source tool other than basic packages such as numpy**. 

If you have any question for any section of this project, please feel free to contact me by the following methods:

E-mail: [douyiming@sjtu.edu.cn](mailto:douyiming@sjtu.edu.cn)

Wechat: 18017112986

