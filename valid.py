import collections
import math

import tensorflow as tf
import data
import train

batch_size = 64
num_steps = 30


def predict(net, src_sentence, src_vocab, tgt_vocab, num_steps):
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = tf.constant([len(src_tokens)])
    src_tokens = data.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    enc_X = tf.expand_dims(src_tokens, axis=0)  # Add the batch axis
    enc_outputs = net.encoder(enc_X, enc_valid_len, training=False)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    dec_X = tf.expand_dims(tf.constant([tgt_vocab['<bos>']]), axis=0)
    output_seq = []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state, training=False)
        dec_X = tf.argmax(Y, axis=2)
        pred = tf.squeeze(dec_X, axis=0)
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred.numpy())
    return ''.join(
        tgt_vocab.to_tokens(tf.reshape(output_seq, shape=-1).numpy().tolist())
    )


def bleu(pred_seq, label_seq, k):
    pred_tokens, label_tokens = list(pred_seq), list(label_seq)
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i:i+n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i:i+n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i:i+n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def valid():
    train_iter, src_vocab, tgt_vocab = data.load_data_cmn(batch_size, num_steps)
    net = train.get_net()
    net.load_weights("model/transformer.param")
    engs = ["They got here yesterday .", "He told the students to be quiet ."]
    chis = ["他们昨天到这里的。", "他告诉了学生要安静."]
    for eng, chi in zip(engs, chis):
        translation = predict(net, eng, src_vocab, tgt_vocab, num_steps)
        print(f'{eng} => {translation}, ',
              f'bleu {bleu(translation, chi, k=2):.3f}')

