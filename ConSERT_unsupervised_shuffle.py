'''
    基于ConSERT无监督的中文相似度模型
    endpoint建模方式，见 https://keras.io/examples/keras_recipes/endpoint_layer_pattern/
    论文：ConSERT: A Contrastive Framework for Self-Supervised Sentence Representation Transfer
    对posid进行shuffle
'''
import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Dense, Embedding, LayerNormalization, Dropout, GlobalAvgPool1D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer
import pickle

import os

from FuncUtils import gelu, softmax, create_initializer, checkpoint_loader, get_cos_distance

from TFDataUtils import single_example_parser_SimCSE_endpoint, batched_data

PARAMS_bert_path = "pretrained/chinese_roberta_wwm_ext_L-12_H-768_A-12"

PARAMS_maxword = 512
PARAMS_vocab_size = 21128
PARAMS_type_vocab_size = 2

PARAMS_head = 12
PARAMS_hidden_size = 768
PARAMS_intermediate_size = 4 * 768
PARAMS_batch_size = 16

PARAMS_mode = "predict"
PARAMS_epochs = 3
PARAMS_lr = 1.0e-5

PARAMS_train_file = [
    'data/TFRecordFiles/simcse_train_lcqmc.tfrecord',
]

PARAMS_dev_file = [
    'data/TFRecordFiles/lcqmc_test.tfrecord',
]

PARAMS_model = "ConSERT_lcqmc_shuffle"
PARAMS_check = "modelfiles/" + PARAMS_model

PARAMS_drop_rate = 0.1


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


class Aug(Layer):
    def __init__(self, **kwargs):
        super(Aug, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        if PARAMS_mode.startswith("train"):  # train stage
            return inputs
        else:  # predict stage
            x, y = inputs
            batchsize = tf.shape(x)[0]
            lx = tf.shape(x)[1]
            ly = tf.shape(y)[1]
            lmax = tf.maximum(lx, ly)
            xx = tf.concat([x, tf.zeros([batchsize, lmax - lx], tf.int32)], axis=-1)
            yy = tf.concat([y, tf.zeros([batchsize, lmax - ly], tf.int32)], axis=-1)

            return tf.concat([xx, yy], axis=0)


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

    def call(self, sen, **kwargs):
        seqlen = tf.shape(sen)[1]
        sen_embed = self.word_embeddings(sen)

        token_embed = self.token_embeddings(tf.zeros_like(sen, dtype=tf.int32))

        if PARAMS_mode.startswith("train"):  # train stage
            pos_embed1 = self.position_embeddings[:seqlen]
            pos_embed2 = tf.gather(pos_embed1, tf.random.shuffle(tf.range(seqlen)))

            all_embed1 = sen_embed + token_embed + pos_embed1
            all_embed2 = sen_embed + token_embed + pos_embed2

            all_embed = tf.concat([all_embed1, all_embed2], axis=0)

            sequencemask = tf.greater(tf.concat([sen, sen], axis=0), 0)
            mask = tf.tile(tf.expand_dims(sequencemask, axis=1), [PARAMS_head, seqlen, 1])

            return self.layernormanddrop(all_embed), mask, sequencemask

        else:  # predict stage
            pos_embed = self.position_embeddings[:seqlen]

            all_embed = sen_embed + token_embed + pos_embed

            sequencemask = tf.greater(sen, 0)
            mask = tf.tile(tf.expand_dims(sequencemask, axis=1), [PARAMS_head, seqlen, 1])

            return self.layernormanddrop(all_embed), mask, sequencemask


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
        self.globalaveragepool1d = GlobalAvgPool1D()

    def call(self, inputs, **kwargs):
        x, mask = inputs

        return self.globalaveragepool1d(x, mask)


class MyLoss(Layer):
    def __init__(self, **kwargs):
        super(MyLoss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        y_pred = tf.nn.l2_normalize(inputs, axis=-1)
        y_pred, y_predplus = tf.split(y_pred, 2)

        batch_size = tf.shape(y_pred)[0]

        zero = tf.zeros([batch_size, batch_size], tf.int32)
        one = tf.eye(num_rows=batch_size, num_columns=batch_size, dtype=tf.int32)

        cos = tf.reduce_sum(y_pred * y_predplus, axis=-1)
        cos = (1.0 + cos) / 2.0

        similarity11 = tf.matmul(y_pred, y_pred, transpose_b=True)
        similarity12 = tf.matmul(y_pred, y_predplus, transpose_b=True)
        similarity21 = tf.matmul(y_predplus, y_pred, transpose_b=True)
        similarity22 = tf.matmul(y_predplus, y_predplus, transpose_b=True)
        similarity1 = tf.concat([similarity11, similarity12], axis=1)
        similarity2 = tf.concat([similarity21, similarity22], axis=1)
        similarity = tf.concat([similarity1, similarity2], axis=0)

        similarity += (1. - tf.pow(2., 31.)) * tf.eye(num_rows=2 * batch_size, num_columns=2 * batch_size,
                                                      dtype=tf.float32)

        tao = 0.1
        similarity = similarity / tao

        y_true1 = tf.concat([zero, one], axis=1)
        y_true2 = tf.concat([one, zero], axis=1)
        y_true = tf.concat([y_true1, y_true2], axis=0)

        loss = tf.reduce_mean(categorical_crossentropy(y_true, similarity, from_logits=True))
        self.add_loss(loss)

        return cos


class USER():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    def build_model_train(self):
        sen = Input(shape=[None], name='sen', dtype=tf.int32)

        now = Aug(name="aug")(sen)

        now, mask, sequencemask = Embeddings(name="embeddings")(now)

        now = Encoder(layers=12, name="encoder")(inputs=(now, mask))

        now = CLSPool(name="clspool")(inputs=(now, sequencemask))

        cos = MyLoss(name="myloss")(now)

        model = Model(inputs=sen, outputs=cos)

        # tf.keras.utils.plot_model(model, to_file="ConSERT_train_unsupervised.jpg", show_shapes=True, dpi=900)

        model.summary(line_length=200)
        for tv in model.variables:
            print(tv.name, tv.shape)

        return model

    def build_model_predict(self):
        sena = Input(shape=[None], name='sena', dtype=tf.int32)
        senb = Input(shape=[None], name='senb', dtype=tf.int32)

        now = Aug(name="aug")(inputs=(sena, senb))

        now, mask, sequencemask = Embeddings(name="embeddings")(now)

        now = Encoder(layers=12, name="encoder")(inputs=(now, mask))

        now = CLSPool(name="clspool")(inputs=(now, sequencemask))

        cos = MyLoss(name="myloss")(now)

        model = Model(inputs=[sena, senb], outputs=cos)
        # tf.keras.utils.plot_model(model, to_file="ConSERT_predict_unsupervised.jpg", show_shapes=True, dpi=900)

        model.summary(line_length=200)
        for tv in model.variables:
            print(tv.name, tv.shape)

        return model

    def train(self):
        model = self.build_model_train()

        load_model_weights_from_checkpoint_bert(model, PARAMS_bert_path + "/bert_model.ckpt")

        optimizer = Adam(PARAMS_lr)
        model.compile(optimizer=optimizer)

        model.save(PARAMS_check + '/ConSERT.h5')

        train_batch = batched_data(PARAMS_train_file,
                                   single_example_parser_SimCSE_endpoint,
                                   PARAMS_batch_size,
                                   padded_shapes={"sen": [-1]})
        history = model.fit(
            train_batch,
            epochs=PARAMS_epochs,
        )
        model.save(PARAMS_check + "/ConSERT.h5")

        file = open(PARAMS_check + '/history.pkl', 'wb')
        pickle.dump(history.history, file)
        file.close()

    def predict(self, sentencesa, sentencesb):
        sen2ida = self.tokenizer(sentencesa, padding=True, return_tensors="tf")["input_ids"]
        sen2idb = self.tokenizer(sentencesb, padding=True, return_tensors="tf")["input_ids"]

        model = self.build_model_predict()
        model.load_weights(PARAMS_check + "/ConSERT.h5", by_name=True)

        result = model.predict([sen2ida, sen2idb])
        for i in range(len(sentencesa)):
            print(sentencesa[i])
            print(sentencesb[i])
            print("相似度: ", result[i], "\n")


if __name__ == "__main__":
    if not os.path.exists(PARAMS_check):
        os.makedirs(PARAMS_check)
    user = USER()

    if PARAMS_mode.startswith('train'):
        user.train()

    elif PARAMS_mode == "predict":
        user.predict([
            "微信号怎么二次修改",
            "云赚钱怎么样",
            "我喜欢北京",
            "我喜欢晨跑",
            "你是谁？"
        ],
            [
                "怎么再二次修改微信号",
                "怎么才能赚钱",
                "我不喜欢北京",
                "北京是中国的首都",
                "武汉市长江大桥"
            ])
