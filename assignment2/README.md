# Assignment 2: HMM Model

In this assignment, we are required to split the Chinese sentence using the trained HMM model, and compute the probability of the sentence using both the forward and backward algorithm.

## 1. Split Chinese Sentence by Viterbi Algorithm

### 1.1 Basic Idea

The basic idea of Viterbi algorithm is quite similar to Dynamic-Programming:

<img src="D:\OneDrive - sjtu.edu.cn\大三上\自然语言处理\Projects&Assignments\Assignment2\README.assets\image-20211027142236731.png" alt="image-20211027142236731" style="zoom:67%;" />

Every value can be simply obtained by the three matrices: start_prob, trans_mat and emission_mat.

### 1.2 Result

```
Before split: 窦铱明是个好同学
After split: ['窦铱', '明是', '个', '好', '同学']
```

## 2. Compute Probability by Forward&Backward Algorithm

### 2.1 Basic Idea

Instead of finding the maximum of the candidates, the forward and backward algorithms computes the sum of the values:

##### Forward Algorithm

<img src="D:\OneDrive - sjtu.edu.cn\大三上\自然语言处理\Projects&Assignments\Assignment2\README.assets\image-20211027142716193.png" alt="image-20211027142716193" style="zoom:67%;" />

##### Backward Algorithm

<img src="D:\OneDrive - sjtu.edu.cn\大三上\自然语言处理\Projects&Assignments\Assignment2\README.assets\image-20211027142736774.png" alt="image-20211027142736774" style="zoom:67%;" />

### 2.2 Result

The results of forward and backward algorithm are the same:

```
Prob computed by forward algorithm: 3.9499806889177e-29
Prob computed by backward algorithm: 3.9499806889177e-29
```

## Appendix

The code of each part:

```python
import pickle
import numpy as np


def str2no(str):
    return [ord(i) for i in str]


def no2str(str):
    return ''.join([chr(i) for i in str])


class HMM:
    def __init__(self, param_path):
        with open(param_path, 'rb') as f:
            self.params = pickle.load(f)

    def viterbi(self, test_str):
        test_str = str2no(test_str)
        l = len(test_str)
        dp = np.zeros((2, l))
        path = np.zeros((2, l))
        # start prob
        dp[0][0] = self.params['start_prob'][0] *\
            self.params['emission_mat'][0][test_str[0]]
        dp[1][0] = self.params['start_prob'][1] *\
            self.params['emission_mat'][1][test_str[0]]
        # compute dp
        for i in range(1, l):
            for j in range(2):
                if dp[0][i-1] * self.params['trans_mat'][0][j] * self.params['emission_mat'][j][test_str[i]] >\
                        dp[1][i-1] * self.params['trans_mat'][1][j] * self.params['emission_mat'][j][test_str[i]]:
                    dp[j][i] = dp[0][i-1] * self.params['trans_mat'][0][j] * \
                        self.params['emission_mat'][j][test_str[i]]
                    path[j][i] = 0
                else:
                    dp[j][i] = dp[1][i-1] * self.params['trans_mat'][1][j] * \
                        self.params['emission_mat'][j][test_str[i]]
                    path[j][i] = 1
        labels = [0] if dp[0][l-1] > dp[1][l-1] else [1]
        for i in range(l-1, 0, -1):
            labels.insert(0, int(path[labels[0]][i]))
        labels = np.array(labels)
        # split
        cut = np.where(labels == 1)[0]
        test_str = no2str(test_str)
        ans = [test_str[0:cut[0]+1]]
        for i in range(1, cut.shape[0]):
            ans.append(test_str[cut[i-1]+1:cut[i]+1])
        if not cut[-1] == l-1:
            ans.append(test_str[cut[-1]+1:])
        return ans

    def forward(self, test_str):
        test_str = str2no(test_str)
        l = len(test_str)
        dp = np.zeros((2, l))
        # start prob
        dp[0][0] = self.params['start_prob'][0] *\
            self.params['emission_mat'][0][test_str[0]]
        dp[1][0] = self.params['start_prob'][1] *\
            self.params['emission_mat'][1][test_str[0]]
        for i in range(1, l):
            for j in range(2):
                dp[j][i] = sum([
                    dp[0][i-1] * self.params['trans_mat'][0][j] *
                    self.params['emission_mat'][j][test_str[i]],
                    dp[1][i-1] * self.params['trans_mat'][1][j] *
                    self.params['emission_mat'][j][test_str[i]]]
                )
        return dp[0][l-1] + dp[1][l-1]

    def backward(self, test_str):
        test_str = str2no(test_str)
        l = len(test_str)
        dp = np.zeros((2, l))
        # start prob
        dp[0][l-1] = 1
        dp[1][l-1] = 1
        for i in range(l - 2, -1, -1):
            for j in range(2):
                dp[j][i] = sum([
                    dp[0][i + 1] * self.params['trans_mat'][j][0] *
                    self.params['emission_mat'][0][test_str[i + 1]],
                    dp[1][i + 1] * self.params['trans_mat'][j][1] *
                    self.params['emission_mat'][1][test_str[i + 1]]]
                )
        return dp[0][0] * self.params['start_prob'][0] * \
            self.params['emission_mat'][0][test_str[0]] +\
            dp[1][0] * self.params['start_prob'][1] * \
            self.params['emission_mat'][1][test_str[0]]
            
def main(test_str, param_path):
    model = HMM(param_path)
    # Split
    split_str = model.viterbi(test_str)
    print("Before split: {}\nAfter split: {}".format(test_str, split_str))
    # forward
    forward_prob=model.forward(test_str)
    print("Prob computed by forward algorithm: {}".format(forward_prob))
    # backward
    backward_prob=model.backward(test_str)
    print("Prob computed by backward algorithm: {}".format(backward_prob))

if __name__ == '__main__':
    param_path = './trained_model/hmm_parameters.pkl'
    test_str = '窦铱明是个好同学'
    main(test_str, param_path)
```

If you have any question of the result or code, please feel free to contact me:

E-mail: douyiming@sjtu.edu.cn

Wechat: 18017112986