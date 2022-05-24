import collections
import re
import tensorflow as tf
from d2l import tensorflow as d2l

def read_data_cmn():
    with open("cmn.txt", "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
        lines = [line.split("\t")[:2] for line in lines if len(line.strip()) > 0]
        print(f"read 'cmn.txt', {len(lines)} pairs")
        return lines

def preprocess_cmn(lines):
    # 标点不应该成为二者不同的障碍，将所有中文标点替换为英文
    # 此外，为了区分标点和单词，在英文的每个标点前后都加一个空格(除了I'd like to这种)
    table = {ord(f): ord(t) for f, t in zip(
        u'，。！？【】（）％＃＠＆１２３４５６７８９０“”‘’\u202f\xa0',
        u',.!?[]()%#@&1234567890""\'\'  ')}
    for i, line in enumerate(lines):
        for j, token in enumerate(line):
            if j == 0:
                # 每个标点符号前后添加空格
                token = re.sub(r'[,.!?"]', r' \g<0> ', token)
                # 英文数字类似6,000这样的，不加空格
                token = re.sub(r'(\d) (,) (\d+)', r'\g<1>\g<2>\g<3>', token)
                # 多个连续的空格替换为1个
                token = re.sub(' +', ' ', token.strip())
                # 全部小写
                lines[i][j] = token.lower()
            else:
                # 中文标点、空格转英文
                token = token.translate(table)
                # 去除所有空格
                lines[i][j] = re.sub(' ', '', token)
    return lines

def tokenize_cmn(lines, num_examples=None):
    # 将句子转化为token
    source, target = [], []
    for i, line in enumerate(lines):
        if num_examples and i > num_examples:
            break
        source.append(line[0].split(' '))
        target.append(list(line[1]))
    print("convert read lines to tokens")
    return source, target

class Vocab:
    """编码，将token编码为序号数字，
    并定义tokens和index之间互相转换"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        if isinstance(tokens[0], list):  # [[句子1的词], [...]]
            tokens = [token for line in tokens for token in line]
        # 统计每个词的出现次数，counter为(token, count)
        counter = collections.Counter(tokens)
        # 按照词的出现次数排序，出现次数多的在前面
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[indice] for indice in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freq(self):
        return self._token_freqs

def build_array_cmn(lines, vocab, num_steps):
    """将所有lines的token都转为vocab数字，并填充到num_steps，
        返回转化后的array，和每句有效（非填充）的长度
        注意：特殊处理是句子结尾加了个<eos>"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = tf.constant([truncate_pad(l, num_steps, vocab['<pad>'])
                         for l in lines])
    valid_len = tf.reduce_sum(tf.cast(array != vocab['<pad>'], tf.int32), 1)
    return array, valid_len

def truncate_pad(line, num_steps, padding_token):
    """如果句子长度>时间步，截断，否则填充"""
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))

def load_array(data_arrays, batch_size, is_train=True):
    """生成tensorflow训练数据"""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    data_iter = dataset.batch(batch_size)
    return data_iter


def load_data_cmn(batch_size, num_steps, num_examples=None):
    """调用上述方法，读中英翻译数据集，生成数据迭代器"""
    time = d2l.Timer()
    lines = read_data_cmn()
    lines = preprocess_cmn(lines)
    source, target = tokenize_cmn(lines, num_examples)
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_cmn(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_cmn(target, tgt_vocab, num_steps)
    print(f"pad every token to length {num_steps}")
    data_array = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_array, batch_size)
    print(f"load data finished, cost {time.stop():.2f}s")
    return data_iter, src_vocab, tgt_vocab


# 画图查看每个句子的长度的频率
def draw_len_freq():
    lines = read_data_cmn()
    lines = preprocess_cmn(lines)
    source, target = tokenize_cmn(lines)
    datas = source + target
    datas = [len(d) for d in datas]
    counters = collections.Counter(datas)
    sorted_counters = sorted(counters.items(),
                           key=lambda x:x[0])
    X, Y = [], []
    for length, frequency in sorted_counters:
        X.append(length)
        Y.append(frequency)
    d2l.plot(X, Y, xlabel="token length", ylabel="frequency",
             legend=["token length frequency"])
    d2l.plt.show()

# draw_len_freq()
load_data_cmn(32, 30)