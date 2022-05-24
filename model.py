import tensorflow as tf
import numpy as np

class PositionWiseFFN(tf.keras.layers.Layer):
    """基于位置的前馈神经网络，对序列中的所有位置表示应用一个MLP"""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(ffn_num_hiddens)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(ffn_num_outputs)

    def call(self, inputs, **kwargs):
        """
        :param inputs: 输入形状(batch_size, num_steps, num_hiddens)
        :param kwargs: None
        :return: 输出形状(batch_size, num_steps, ffn_num_outputs)
        """
        return self.dense2(self.relu(self.dense1(inputs)))

class AddNorm(tf.keras.layers.Layer):
    """残差连接和层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(normalized_shape)

    def call(self, X, Y, **kwargs):
        return self.ln(self.dropout(Y) + X)

class DotProductAttention(tf.keras.layers.Layer):
    """点积注意力计算法"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, queries, keys, values, valid_lens, **kwargs):
        """下面的形状特指本项目，不具有通用性，原因在于多头注意力的调用
        :param queries: (batch_size*num_heads, num of query, num_hiddens/num_heads)
        :param keys: (batch_size*num_heads, num of k-v pair, num_hiddens/num_heads)
        :param values: (batch_size*num_heads, num of k-v pair, num_hiddens/num_heads)
        :param valid_lens: (num_heads*batch_size, )
        :param kwargs:
        :return: (batch_size*num_heads, num of query, num_hiddens/num_heads)
        """
        d = queries.shape[-1]
        scores = tf.matmul(queries, keys, transpose_b=True) / \
            tf.math.sqrt(tf.cast(d, dtype=tf.float32))
        # 掩掉<pad>填充的部分，这部分不计算注意力
        self.attention_weights = masked_softmax(scores, valid_lens)
        return tf.matmul(self.dropout(self.attention_weights, **kwargs), values)

def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return tf.nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if len(valid_lens.shape) == 1: # 走这个分支
            valid_lens = tf.repeat(valid_lens, repeats=shape[1])
        else:
            valid_lens = tf.reshape(valid_lens, shape=-1)
        X = sequence_mask(tf.reshape(X, shape=(-1, shape[-1])), valid_lens, value=-1e6)
        return tf.nn.softmax(tf.reshape(X, shape=shape), axis=-1)

def sequence_mask(X, valid_lens, value=0):
    maxlen = X.shape[1]
    mask = tf.range(start=0, limit=maxlen, dtype=tf.float32)[None, :] < \
           tf.cast(valid_lens[:, None], tf.float32)
    if len(X.shape) == 3:
        return tf.where(tf.expand_dims(mask, axis=-1), X, value)
    else:  # 走这个分支
        return tf.where(mask, X, value)

class MultiHeadAttention(tf.keras.layers.Layer):
    """多头注意力计算法"""
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads,
                 dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_v = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_o = tf.keras.layers.Dense(num_hiddens, use_bias=bias)

    def call(self, queries, keys, values, valid_lens, **kwargs):
        """
        :param queries: (batch_size, num of queries, num_hiddens)
        :param keys: (batch_size, k-v pair num, num_hiddens)
        :param values: (batch_size, k-v pair num, num_hiddens)
        :param valid_lens: (batch_size, )
        :param kwargs:
        :return: (batch_size, num of queries, num_hiddens)
        """
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            valid_lens = tf.repeat(valid_lens, repeats=self.num_heads, axis=0)
        output = self.attention(queries, keys, values, valid_lens, **kwargs)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

def transpose_qkv(X, num_heads):
    """
    为了方便并行计算而进行的reshape操作
    :param X: (batch_size, num of queries or k-v pair, num_hiddens)
    :param num_heads:
    :return: (batch_size*num_heads, num of queries or k-v pair, num_hiddens/num_heads)
    """
    X = tf.reshape(X, shape=(X.shape[0], X.shape[1], num_heads, -1))
    X = tf.transpose(X, perm=(0,2,1,3))
    X = tf.reshape(X, shape=(-1, X.shape[2], X.shape[3]))
    return X

def transpose_output(X, num_heads):
    """transpose_qkv的逆操作"""
    X = tf.reshape(X, shape=(-1, num_heads, X.shape[1], X.shape[2]))
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    X = tf.reshape(X, shape=(X.shape[0], X.shape[1], -1))
    return X


class EncoderBlock(tf.keras.layers.Layer):
    """transformer编码器"""
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape,
                 ffn_num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size,
                                                num_hiddens, num_heads, dropout, bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def call(self, X, valid_lens, **kwargs):
        """
        :param X: (batch_size, num_steps, num_hiddens)
        :param valid_lens: (batch_size, )
        :param kwargs:
        :return: (batch_size, num_steps, num_hiddens)
        """
        # 自注意力
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens, **kwargs), **kwargs)
        return self.addnorm2(Y, self.ffn(Y), **kwargs)

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(max_len, dtype=np.float32).reshape(-1, 1) / \
            np.power(10000, np.arange(0, num_hiddens, 2, dtype=np.float32) / num_hiddens)
        self.P[:,:,0::2] = np.sin(X)
        self.P[:,:,1::2] = np.cos(X)
    def call(self, X, **kwargs):
        X = X + self.P[:,:X.shape[1], :]
        return self.dropout(X, **kwargs)

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_hiddens, num_heads, num_layers,
                 dropout, bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = [EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                  norm_shape, ffn_num_hiddens, num_heads, dropout,
                                  bias) for _ in range(num_layers)]
    def call(self, X, valid_lens, **kwargs):
        """
        :param X: (batch_size, num_steps)
        :param valid_lens: (batch_size, )
        :param kwargs:
        :return: (batch_size, num_steps, num_hiddens)
        """
        # 因为位置编码值在-1和1之间，因此嵌入值乘以嵌入维度的平方根进行缩放，防止位置编码值过大影响嵌入值
        # 然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * tf.math.sqrt(
            tf.cast(self.num_hiddens, dtype=tf.float32)), **kwargs)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens, **kwargs)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_hiddens, num_heads, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size,
                                                 num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size,
                                                 num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def call(self, X, state, **kwargs):
        """
        :param X: (batch_size, num_steps, num_hiddens)
        :param state: [enc_outputs, enc_valid_lens, record]，
        分别为(batch_size, num_steps(in), num_hiddens), (batch_size, ),
        (num_layers, batch_size, n*num_steps, num_hiddens)，n为连接的词元
        :param kwargs:
        :return: [dec_outputs, state]，其中dec_outputs: (batch_size, num_steps, num_hiddens)
        """
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = tf.concat((state[2][self.i], X), axis=1)  # 将输入X逐渐连接起来
        state[2][self.i] = key_values
        if kwargs["training"]:  # 如果训练，在预测下一个时掩蔽掉下一个
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = tf.repeat(tf.reshape(tf.range(1, num_steps+1),
                                                  shape=(-1, num_steps)),
                                       repeats=batch_size, axis=0)
        else:
            dec_valid_lens = 0
        # 自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens, **kwargs)
        Y = self.addnorm1(X, X2, **kwargs)
        # 编码器-解码器注意力
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens, **kwargs)
        Z = self.addnorm2(Y, Y2, **kwargs)
        return self.addnorm3(Z, self.ffn(Z), **kwargs), state



class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_hiddens, num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = [DecoderBlock(key_size, query_size, value_size, num_hiddens,
                                  norm_shape, ffn_num_hiddens, num_heads, dropout,
                                  i) for i in range(num_layers)]
        self.dense = tf.keras.layers.Dense(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def call(self, X, state, **kwargs):
        """
        :param X:  (batch_size, num_steps)
        :param state: (batch_size, )
        :param kwargs:
        :return:
        """
        X = self.pos_encoding(self.embedding(X) * tf.math.sqrt(
            tf.cast(self.num_hiddens, dtype=tf.float32)), **kwargs)
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)] # 解码器中2个注意力层
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state, **kwargs)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


class EncoderDecoder(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    def call(self, enc_X, dec_X, *args, **kwargs):
        enc_outputs = self.encoder(enc_X, *args, **kwargs)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state, **kwargs)