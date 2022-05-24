import data
import model
from d2l import tensorflow as d2l
import tensorflow as tf

# 超参数
num_hiddens = 32
num_layers = 2
dropout = 0.1
batch_size = 64
num_steps = 30
lr = 0.005
num_epochs = 200
device = d2l.try_gpu()
ffn_num_hiddens = 64
num_heads = 4
key_size = 32
query_size = 32
value_size = 32
norm_shape = [2]

def grad_clipping(grads, theta):
    """梯度裁剪"""
    theta = tf.constant(theta, dtype=tf.float32)
    new_grad = []
    for grad in grads:
        if isinstance(grad, tf.IndexedSlices):
            new_grad.append(tf.convert_to_tensor(grad))
        else:
            new_grad.append(grad)
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad**2)).numpy() for grad in new_grad))
    norm = tf.cast(norm, tf.float32)
    if tf.greater(norm, theta):
        for i, grad in enumerate(new_grad):
            new_grad[i] = grad * theta / norm
    return new_grad



class MaskedSoftmaxCELoss(tf.keras.losses.Loss):
    def __init__(self, valid_len):
        super(MaskedSoftmaxCELoss, self).__init__(reduction="none")
        self.valid_len = valid_len
    def call(self, y_true, y_pred):
        weights = tf.ones_like(y_true, dtype=tf.float32)
        weights = model.sequence_mask(weights, self.valid_len)
        y_true_one_hot = tf.one_hot(y_true, depth=y_pred.shape[-1])
        unweighted_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction="none")(y_true_one_hot, y_pred)
        weighted_loss = tf.reduce_mean((unweighted_loss * weights), axis=1)
        return weighted_loss


def train_model_transformer(net, data_iter, lr, num_epochs, tgt_vocab, device):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    animator = d2l.Animator(xlabel="epoch", ylabel="loss", xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, num of tokens
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x for x in batch]
            bos = tf.reshape(tf.constant([tgt_vocab['<bos>']] * Y.shape[0]), shape=(-1, 1))
            dec_input = tf.concat([bos, Y[:,:-1]], axis=1)
            with tf.GradientTape() as tape:
                Y_hat, _ = net(X, dec_input, X_valid_len, training=True)
                l = MaskedSoftmaxCELoss(Y_valid_len)(Y, Y_hat)
            gradients = tape.gradient(l, net.trainable_variables)
            gradients = grad_clipping(gradients, 1)
            optimizer.apply_gradients(zip(gradients, net.trainable_variables))
            num_tokens = tf.reduce_sum(Y_valid_len).numpy()
            metric.add(tf.reduce_sum(l), num_tokens)
        print(f"train epoch {epoch+1} finish.")
        if (epoch+1) % 10 == 0:
            animator.add(epoch+1, (metric[0]/metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')
    d2l.plt.show()


def train():
    train_iter, src_vocab, tgt_vocab = data.load_data_cmn(batch_size, num_steps)
    encoder = model.TransformerEncoder(vocab_size=len(src_vocab), key_size=key_size, query_size=query_size,
                                       value_size=value_size, num_hiddens=num_hiddens, norm_shape=norm_shape,
                                       ffn_num_hiddens=ffn_num_hiddens, num_heads=num_heads,
                                       num_layers=num_layers, dropout=dropout)
    decoder = model.TransformerDecoder(vocab_size=len(tgt_vocab), key_size=key_size, query_size=query_size,
                                       value_size=value_size, num_hiddens=num_hiddens, norm_shape=norm_shape,
                                       ffn_num_hiddens=ffn_num_hiddens, num_heads=num_heads,
                                       num_layers=num_layers, dropout=dropout)
    net = model.EncoderDecoder(encoder, decoder)
    train_model_transformer(net, train_iter, lr, num_epochs, tgt_vocab, device)
    net.save_weights("model/transformer.param")


def get_net():
    train_iter, src_vocab, tgt_vocab = data.load_data_cmn(batch_size, num_steps)
    encoder = model.TransformerEncoder(vocab_size=len(src_vocab), key_size=key_size, query_size=query_size,
                                       value_size=value_size, num_hiddens=num_hiddens, norm_shape=norm_shape,
                                       ffn_num_hiddens=ffn_num_hiddens, num_heads=num_heads,
                                       num_layers=num_layers, dropout=dropout)
    decoder = model.TransformerDecoder(vocab_size=len(tgt_vocab), key_size=key_size, query_size=query_size,
                                       value_size=value_size, num_hiddens=num_hiddens, norm_shape=norm_shape,
                                       ffn_num_hiddens=ffn_num_hiddens, num_heads=num_heads,
                                       num_layers=num_layers, dropout=dropout)
    net = model.EncoderDecoder(encoder, decoder)
    return net



