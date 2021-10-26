# compute PPL for pre-trained n-gram model
# For any question or problem, please feel free to contact:
# Email: douyiming@sjtu.edu.cn
# Wechat: 18017112986

# load model
model_path = './data/cs382_1.arpa'
data = {}
with open(model_path, 'r') as f:
    raw_data = f.readlines()
    data['uni'] = raw_data[7:19]
    data['bi'] = raw_data[21:98]
    data['tri'] = raw_data[100:142]
for k, v in data.items():
    data[k] = [s.split() for s in v]
seq2info = {}  # {seq: {"log_p": xxx, "backoff": xxx}}
for k, v in data.items():
    for l in v:
        if k == 'uni':
            seq = (l[1])
            info = {"log_p": float(l[0])}
            info["backoff"] = float(l[2]) if len(l) == 3 else 0
            seq2info[seq] = info
        elif k == 'bi':
            seq = (l[1], l[2])
            info = {"log_p": float(l[0])}
            info["backoff"] = float(l[3]) if len(l) == 4 else 0
            seq2info[seq] = info
        elif k == 'tri':
            seq = (l[1], l[2], l[3])
            info = {"log_p": float(l[0])}
            info["backoff"] = float(l[4]) if len(l) == 5 else 0
            seq2info[seq] = info


def p(seq):
    print(tuple(seq))
    l = len(seq)
    assert 1 <= l and l <= 3
    if l == 1:
        info = seq2info.get(seq)
    elif l == 2:
        info = seq2info.get(seq)
        if info == None:  # unseen
            return p(seq[1])
    else:
        info = seq2info.get(seq)
        if info == None:  # unseen
            return p(seq[1:])
    return info['log_p']+info['backoff']


def ppl(seq):
    # computes the perplexity of a sequence
    seq_ex = ['<s>']+[s for s in seq]+['</s>']
    ans = p(tuple(seq_ex[0:2]))
    for i in range(0, len(seq_ex)-2):
        ans += p(tuple(seq_ex[i:i+3]))
    print(ans)
    ans = pow(10, -1/len(seq_ex)*ans)
    return ans


seq = ['021033210023', '019033910051', '120033910006', '120033910013']
ans = {}
for s in seq:
    print("computing perplexity of {}...".format(s))
    print("PPL = {}".format(ppl(s)))