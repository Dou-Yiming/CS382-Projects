from models.HMM import HMM


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
