'''
    基于ESimCSE无监督的中文相似度模型
    endpoint建模方式，见 https://keras.io/examples/keras_recipes/endpoint_layer_pattern/
    训练模型和推理模型统一
'''
import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Dense, Embedding, LayerNormalization, Dropout
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from transformers import BertTokenizer
from scipy.stats import spearmanr
import numpy as np
import os

from FuncUtils import gelu, softmax, create_initializer, checkpoint_loader

from TFDataUtils import batched_data

PARAMS_bert_path = "pretrained/chinese_roberta_wwm_ext_L-12_H-768_A-12"

PARAMS_maxword = 512
PARAMS_vocab_size = 21128
PARAMS_type_vocab_size = 2

PARAMS_head = 12
PARAMS_hidden_size = 768
PARAMS_intermediate_size = 4 * 768
PARAMS_batch_size = 16

PARAMS_mode = "train"
PARAMS_epochs = 2
PARAMS_lr = 1.0e-5

PARAMS_train_file = [
    'data/TFRecordFiles/lcqmc_train.tfrecord',
]

PARAMS_dev_file = [
    'data/TFRecordFiles/lcqmc_test.tfrecord',
]

PARAMS_model = "ESimCSE_lcqmc_endpoints"
PARAMS_check = "modelfiles/" + PARAMS_model

PARAMS_drop_rate = 0.3


def load_model_weights_from_checkpoint_bert(model, checkpoint_file):
    """Load trained official modelfiles from checkpoint.

    :param model: Built keras modelfiles.
    :param checkpoint_file: The path to the checkpoint files, should end with '.ckpt'.
    """
    loader = checkpoint_loader(checkpoint_file)

    weights = [
        loader('bert/embeddings/position_embeddings'),
        loader('bert/embeddings/word_embeddings'),
        loader('bert/embeddings/token_type_embeddings'),
        loader('bert/embeddings/LayerNorm/gamma'),
        loader('bert/embeddings/LayerNorm/beta'),
    ]
    model.get_layer('embeddings').set_weights(weights)

    weights_a = []
    weights_f = []
    for i in range(12):
        pre = 'bert/encoder/layer_' + str(i) + '/'
        weights_a.extend([
            loader(pre + 'attention/self/query/kernel'),
            loader(pre + 'attention/self/query/bias'),
            loader(pre + 'attention/self/key/kernel'),
            loader(pre + 'attention/self/key/bias'),
            loader(pre + 'attention/self/value/kernel'),
            loader(pre + 'attention/self/value/bias'),
            loader(pre + 'attention/output/dense/kernel'),
            loader(pre + 'attention/output/dense/bias'),
            loader(pre + 'attention/output/LayerNorm/gamma'),
            loader(pre + 'attention/output/LayerNorm/beta')])

        weights_f.extend([
            loader(pre + 'intermediate/dense/kernel'),
            loader(pre + 'intermediate/dense/bias'),
            loader(pre + 'output/dense/kernel'),
            loader(pre + 'output/dense/bias'),
            loader(pre + 'output/LayerNorm/gamma'),
            loader(pre + 'output/LayerNorm/beta')])

    weights = weights_a + weights_f
    model.get_layer('encoder').set_weights(weights)

    # weights = [
    #     loader('bert/pooler/dense/kernel'),
    #     loader('bert/pooler/dense/bias'),
    # ]
    # model.get_layer('pooler').set_weights(weights)


def single_example_parser_train(serialized_example):
    sequence_features = {
        'sena': tf.io.FixedLenSequenceFeature([], tf.int64),
    }

    _, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=serialized_example,
        sequence_features=sequence_features
    )

    sena = sequence_parsed['sena']

    seqlen = tf.shape(sena)[0] - 2

    indx = tf.cast(tf.less(tf.random.uniform([seqlen], 0.0, 1.0), 0.15), tf.int32)

    num = tf.reduce_sum(indx)

    indxr = tf.nn.top_k(indx, num).indices

    a = sena[None, :]

    senb = tf.cond(tf.equal(num, 0),
                   lambda: a,
                   lambda: tf.cond(tf.equal(num, 1),
                                   lambda: tf.concat(
                                       [a[:, :indxr[0] + 2], a[:, indxr[0] + 1:indxr[0] + 2], a[:, indxr[0] + 2:]],
                                       axis=-1),
                                   lambda: tf.cond(tf.equal(num, 2),
                                                   lambda: tf.concat(
                                                       [a[:, :indxr[0] + 2], a[:, indxr[0] + 1:indxr[0] + 2],
                                                        a[:, indxr[0] + 2:indxr[1] + 2],
                                                        a[:, indxr[1] + 1:indxr[1] + 2], a[:, indxr[1] + 2:]],
                                                       axis=-1),
                                                   lambda: tf.concat(
                                                       [a[:, :indxr[0] + 2], a[:, indxr[0] + 1:indxr[0] + 2],
                                                        a[:, indxr[0] + 2:indxr[1] + 2],
                                                        a[:, indxr[1] + 1:indxr[1] + 2],
                                                        a[:, indxr[1] + 2:indxr[2] + 2],
                                                        a[:, indxr[2] + 1:indxr[2] + 2],
                                                        a[:, indxr[2] + 2:]], axis=-1))))
    senb = tf.squeeze(senb, axis=0)

    return {"sena": sena, "senb": senb}


def single_example_parser_dev(serialized_example):
    sequence_features = {
        'sena': tf.io.FixedLenSequenceFeature([], tf.int64),
        'senb': tf.io.FixedLenSequenceFeature([], tf.int64),
    }

    context_features = {
        "label": tf.io.FixedLenFeature([], tf.int64)
    }

    context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    sena = sequence_parsed['sena']
    senb = sequence_parsed['senb']

    label = context_parsed['label']

    return {"sena": sena, "senb": senb, "label": label}


class Aug(Layer):
    def __init__(self, **kwargs):
        super(Aug, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x, y = inputs
        batchsize = tf.shape(x)[0]
        lx = tf.shape(x)[1]
        ly = tf.shape(y)[1]
        lmax = tf.maximum(lx, ly)
        xx = tf.concat([x, tf.zeros([batchsize, lmax - lx], tf.int32)], axis=-1)
        yy = tf.concat([y, tf.zeros([batchsize, lmax - ly], tf.int32)], axis=-1)

        return tf.concat([xx, yy], axis=0)


class Mask(Layer):
    def __init__(self, **kwargs):
        super(Mask, self).__init__(**kwargs)

    def call(self, senwrong, **kwargs):
        sequencemask = tf.greater(senwrong, 0)
        seq_length = tf.shape(senwrong)[1]
        mask = tf.tile(tf.expand_dims(sequencemask, axis=1), [PARAMS_head, seq_length, 1])

        return mask, seq_length


class Embeddings(Layer):
    def __init__(self, **kwargs):
        super(Embeddings, self).__init__(**kwargs)

    def build(self, input_shape):
        self.word_embeddings = Embedding(PARAMS_vocab_size,
                                         PARAMS_hidden_size,
                                         embeddings_initializer=create_initializer(),
                                         dtype=tf.float32,
                                         name="word_embeddings")

        self.token_embeddings = Embedding(PARAMS_type_vocab_size,
                                          PARAMS_hidden_size,
                                          embeddings_initializer=create_initializer(),
                                          dtype=tf.float32,
                                          name='token_type_embeddings')

        self.position_embeddings = self.add_weight(name='position_embeddings',
                                                   shape=[PARAMS_maxword, PARAMS_hidden_size],
                                                   dtype=tf.float32,
                                                   initializer=create_initializer())

        self.layernormanddrop = LayerNormalizeAndDrop(name="layernormanddrop")

        super(Embeddings, self).build(input_shape)

    def call(self, inputs, **kwargs):
        sen, seqlen = inputs
        sen_embed = self.word_embeddings(sen)

        token_embed = self.token_embeddings(tf.zeros_like(sen, dtype=tf.int32))
        pos_embed = self.position_embeddings[:seqlen]

        all_embed = sen_embed + token_embed + pos_embed

        return self.layernormanddrop(all_embed)


class LayerNormalizeAndDrop(Layer):
    def __init__(self, **kwargs):
        super(LayerNormalizeAndDrop, self).__init__(**kwargs)

    def build(self, input_shape):
        self.layernorm = LayerNormalization(name="layernorm")
        self.dropout = Dropout(PARAMS_drop_rate)
        super(LayerNormalizeAndDrop, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.dropout(self.layernorm(inputs))


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense_q = Dense(PARAMS_hidden_size,
                             name='query',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.dense_k = Dense(PARAMS_hidden_size,
                             name='key',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.dense_v = Dense(PARAMS_hidden_size,
                             name='value',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.dense_o = Dense(PARAMS_hidden_size,
                             name='output',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.dropoutsoft = Dropout(PARAMS_drop_rate)
        self.dropoutres = Dropout(PARAMS_drop_rate)
        self.layernorm = LayerNormalization(name='layernormattn')

        super(Attention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x, mask = inputs
        q = tf.concat(tf.split(self.dense_q(x), PARAMS_head, axis=-1), axis=0)
        k = tf.concat(tf.split(self.dense_k(x), PARAMS_head, axis=-1), axis=0)
        v = tf.concat(tf.split(self.dense_v(x), PARAMS_head, axis=-1), axis=0)
        qk = tf.matmul(q, tf.transpose(k, [0, 2, 1])) / tf.sqrt(PARAMS_hidden_size / PARAMS_head)
        attention_output = self.dense_o(tf.concat(
            tf.split(tf.matmul(self.dropoutsoft(softmax(qk, mask)), v), PARAMS_head, axis=0),
            axis=-1))

        return self.layernorm(x + self.dropoutres(attention_output))


class FeedFord(Layer):
    def __init__(self, **kwargs):
        super(FeedFord, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense_ffgelu = Dense(PARAMS_intermediate_size,
                                  kernel_initializer=create_initializer(),
                                  dtype=tf.float32,
                                  name='intermediate',
                                  activation=gelu)
        self.dense_ff = Dense(PARAMS_hidden_size,
                              kernel_initializer=create_initializer(),
                              dtype=tf.float32,
                              name='output')
        self.dropoutres = Dropout(PARAMS_drop_rate)
        self.layernorm = LayerNormalization(name='layernormffd')
        super(FeedFord, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.layernorm(inputs + self.dropoutres(self.dense_ff(self.dense_ffgelu(inputs))))


class Encoder(Layer):
    def __init__(self, layers, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.layers = layers

    def build(self, input_shape):
        self.attention = [Attention(name="attnlayer_%d" % k) for k in range(self.layers)]
        self.ffd = [FeedFord(name="ffdlayer_%d" % k) for k in range(self.layers)]

        super(Encoder, self).build(input_shape)

    def get_config(self):
        config = {"layers": self.layers}
        base_config = super(Encoder, self).get_config()
        return dict(base_config, **config)

    def call(self, inputs, **kwargs):
        x, mask = inputs
        for k in range(self.layers):
            x = self.ffd[k](self.attention[k](inputs=(x, mask)))

        return x


class CLSPool(Layer):
    def __init__(self, **kwargs):
        super(CLSPool, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return inputs[:, 0]


class CLSAFTERPool(Layer):
    def __init__(self, **kwargs):
        super(CLSAFTERPool, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense = Dense(PARAMS_hidden_size, activation=tf.tanh)
        super(CLSAFTERPool, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.dense(inputs[:, 0])


class MyLoss(Layer):
    def __init__(self, **kwargs):
        super(MyLoss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        y_pred = tf.nn.l2_normalize(inputs, axis=-1)
        y_pred, y_predplus = tf.split(y_pred, 2)

        cos = tf.reduce_sum(y_pred * y_predplus, axis=-1)
        cos = (1.0 + cos) / 2.0

        similarity = tf.matmul(y_pred, y_predplus, transpose_b=True)
        similarityplus = tf.matmul(y_predplus, y_pred, transpose_b=True)
        similarity = tf.concat([similarity, similarityplus], axis=0)
        tao = 0.05
        similarity = similarity / tao

        y_true = tf.eye(tf.shape(y_pred)[0], dtype=tf.int32)
        y_true = tf.tile(y_true, [2, 1])

        loss = tf.reduce_mean(categorical_crossentropy(y_true, similarity, from_logits=True))
        self.add_loss(loss)

        return cos


class CheckCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation):
        super(CheckCallback, self).__init__()

        self.validation_data = validation

    def on_train_begin(self, logs=None):
        result = self.model.predict([sen2ida, sen2idb])
        for i in range(len(sentencesa)):
            print(sentencesa[i])
            print(sentencesb[i])
            print("相似度: ", result[i], "\n")

        predictions = []
        labels = []
        for data in self.validation_data:
            res = self.model.predict([data["sena"], data["senb"]])
            predictions.extend(res)
            labels.extend(data["label"].numpy())

        spearmancoff = spearmanr(np.array(predictions), np.array(labels)).correlation

        print("Spearman相关度: ", spearmancoff, "\n")

    def on_epoch_end(self, epoch, logs=None):
        result = self.model.predict([sen2ida, sen2idb])
        for i in range(len(sentencesa)):
            print(sentencesa[i])
            print(sentencesb[i])
            print("相似度: ", result[i], "\n")

        predictions = []
        labels = []
        for data in self.validation_data:
            res = self.model.predict([data["sena"], data["senb"]])
            predictions.extend(res)
            labels.extend(data["label"].numpy())

        spearmancoff = spearmanr(np.array(predictions), np.array(labels)).correlation

        print("Spearman相关度: ", spearmancoff, "\n")

        self.model.save(PARAMS_check + "/SimCSE.h5")


class USER():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    def build_model(self):
        sena = Input(shape=[None], name='sena', dtype=tf.int32)
        senb = Input(shape=[None], name='senb', dtype=tf.int32)

        now = Aug(name="aug")(inputs=(sena, senb))

        mask, seqlen = Mask(name="mask")(now)

        sen_embed = Embeddings(name="embeddings")(inputs=(now, seqlen))

        now = Encoder(layers=12, name="encoder")(inputs=(sen_embed, mask))

        # now = CLSAFTERPool(name="pooler")(now)
        now = CLSPool(name="pooler")(now)

        cos = MyLoss(name="myloss")(now)

        model = Model(inputs=[sena, senb], outputs=cos)

        model.summary(line_length=200)

        for tv in model.variables:
            print(tv.name, tv.shape)

        return model

    def train(self):
        model = self.build_model()

        load_model_weights_from_checkpoint_bert(model, PARAMS_bert_path + "/bert_model.ckpt")

        optimizer = Adam(PARAMS_lr)
        model.compile(optimizer=optimizer)

        train_batch = batched_data(PARAMS_train_file,
                                   single_example_parser_train,
                                   PARAMS_batch_size,
                                   padded_shapes={"sena": [-1], "senb": [-1]},
                                   shuffle=False,
                                   repeat=False)

        dev_batch = batched_data(PARAMS_dev_file,
                                 single_example_parser_dev,
                                 PARAMS_batch_size,
                                 padded_shapes={"sena": [-1], "senb": [-1], "label": []},
                                 shuffle=False,
                                 repeat=False)

        callbacks = [
            # EarlyStopping(monitor='val_loss', patience=7),
            # ModelCheckpoint(filepath=PARAMS_check + '/SimCSE.h5',
            #                 monitor='val_loss',
            #                 save_best_only=True),
            CheckCallback(validation=dev_batch)
        ]

        history = model.fit(
            train_batch,
            epochs=PARAMS_epochs,
            # steps_per_epoch=625,
            callbacks=callbacks
        )

        with open(PARAMS_check + "/history.txt", "w", encoding="utf-8") as fw:
            fw.write(str(history.history))

    def predict(self):
        model = self.build_model()
        model.load_weights(PARAMS_check + "/SimCSE.h5", by_name=True)

        result = model.predict({"sena": sen2ida, "senb": sen2idb})

        for i in range(len(sentencesa)):
            print(sentencesa[i])
            print(sentencesb[i])
            print("相似度: ", result[i], "\n")


if __name__ == "__main__":
    if not os.path.exists(PARAMS_check):
        os.makedirs(PARAMS_check)
    user = USER()
    sentencesa = [
        "微信号怎么二次修改",
        "云赚钱怎么样",
        "我喜欢北京",
        "我喜欢晨跑",
    ]
    sentencesb = [
        "怎么再二次修改微信号",
        "怎么才能赚钱",
        "我不喜欢北京",
        "北京是中国的首都",
    ]
    sen2ida = user.tokenizer(sentencesa, padding=True, return_tensors="tf")["input_ids"]
    sen2idb = user.tokenizer(sentencesb, padding=True, return_tensors="tf")["input_ids"]

    if PARAMS_mode.startswith('train'):
        user.train()

    elif PARAMS_mode == "predict":
        user.predict()
