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
        # start prob
        dp[0][0] = self.params['start_prob'][0] *\
            self.params['emission_mat'][0][test_str[0]]
        dp[1][0] = self.params['start_prob'][1] *\
            self.params['emission_mat'][1][test_str[0]]
        # compute dp
        for i in range(1, l):
            for j in range(2):
                dp[j][i] = max(
                    dp[0][i-1] * self.params['trans_mat'][0][j] *
                    self.params['emission_mat'][j][test_str[i]],
                    dp[1][i-1] * self.params['trans_mat'][1][j] *
                    self.params['emission_mat'][j][test_str[i]]
                )
        # get label
        labels = []
        for i in range(l):
            if dp[0][i] > dp[1][i]:
                labels.append(0)
            else:
                labels.append(1)
        labels = np.array(labels)
        # split
        cut = np.where(labels == 1)[0]
        test_str = no2str(test_str)
        ans = [test_str[0:cut[0]+1]]
        for i in range(1, cut.shape[0]):
            ans.append(test_str[cut[i-1]+1:cut[i]+1])
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