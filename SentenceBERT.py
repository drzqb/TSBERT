'''
    基于SentenceBERT的中文相似度模型
'''
import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Dense, Embedding, LayerNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from official.nlp.optimization import WarmUp, AdamWeightDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from transformers import BertTokenizer
from math import ceil
import pickle

import os

from FuncUtils import gelu, softmax, create_initializer, checkpoint_loader, get_cos_distance

from TFDataUtils import single_example_parser_SBERT, batched_data

PARAMS_bert_path = "pretrained/chinese_roberta_wwm_ext_L-12_H-768_A-12"

PARAMS_maxword = 512
PARAMS_vocab_size = 21128
PARAMS_type_vocab_size = 2

PARAMS_head = 12
PARAMS_hidden_size = 768
PARAMS_intermediate_size = 4 * 768
PARAMS_batch_size = 8

PARAMS_mode = "train0"
PARAMS_epochs = 20
PARAMS_per_save = ceil((238766) / PARAMS_batch_size)
PARAMS_decay_steps = PARAMS_epochs * PARAMS_per_save
PARAMS_warmup_steps = 1 * PARAMS_per_save
PARAMS_lr = 1.0e-5
PARAMS_patience = 3
PARAMS_drop_rate = 0.1

PARAMS_train_file = [
    'data/TFRecordFiles/lcqmc_train.tfrecord',
]

PARAMS_dev_file = [
    'data/TFRecordFiles/lcqmc_dev.tfrecord',
]

PARAMS_model = "SBERT_lcqmc"
PARAMS_check = "modelfiles/" + PARAMS_model


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

        self.layernorm = LayerNormalization(name="layernorm")
        self.dropout = Dropout(PARAMS_drop_rate)

    def call(self, inputs, **kwargs):
        return self.dropout(self.layernorm(inputs))


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

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

    def call(self, inputs, **kwargs):
        return self.layernorm(inputs + self.dropoutres(self.dense_ff(self.dense_ffgelu(inputs))))


class Encoder(Layer):
    def __init__(self, layers, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.layers = layers

        self.attention = [Attention(name="attnlayer_%d" % k) for k in range(self.layers)]
        self.ffd = [FeedFord(name="ffdlayer_%d" % k) for k in range(self.layers)]

    def get_config(self):
        config = {"layers": self.layers}
        base_config = super(Encoder, self).get_config()
        return dict(base_config, **config)

    def call(self, inputs, **kwargs):
        x, mask = inputs
        for k in range(self.layers):
            x = self.ffd[k](self.attention[k](inputs=(x, mask)))

        return x


class GlobalAveragePool(Layer):
    def __init__(self, **kwargs):
        super(GlobalAveragePool, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        seqenceouput_av = tf.reduce_mean(inputs, axis=1)

        return seqenceouput_av


class ProjectTrain(Layer):
    def __init__(self, **kwargs):
        super(ProjectTrain, self).__init__(**kwargs)

        self.projectdense = Dense(1, activation="sigmoid")
        self.dropout = Dropout(PARAMS_drop_rate)

    def call(self, inputs, **kwargs):
        u, v = tf.split(inputs, 2)
        fuse = tf.concat([u, v, tf.abs(u - v)], axis=-1)
        output = self.projectdense(self.dropout(fuse))

        return output


class ProjectPredict(Layer):
    def __init__(self, **kwargs):
        super(ProjectPredict, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        u, v = tf.split(inputs, 2)
        cos = get_cos_distance(u, v)
        return (1.0 + cos) / 2.0


class USER():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    def build_model(self):
        sena = Input(shape=[None], name='sena', dtype=tf.int32)
        senb = Input(shape=[None], name='senb', dtype=tf.int32)

        now = Aug(name="aug")(inputs=(sena, senb))

        mask, seqlen = Mask(name="mask")(now)

        now = Embeddings(name="embeddings")(inputs=(now, seqlen))

        now = Encoder(layers=12, name="encoder")(inputs=(now, mask))

        now = GlobalAveragePool(name="globalaveragepool")(now)

        if PARAMS_mode.startswith("train"):
            logits = ProjectTrain(name="projecttrain")(now)
            model = Model(inputs=[sena, senb], outputs=logits)
            # tf.keras.utils.plot_model(model, to_file="SBERT_train.jpg", show_shapes=True, dpi=900)
        else:
            logits = ProjectPredict(name="projectpredict")(now)
            model = Model(inputs=[sena, senb], outputs=logits)
            # tf.keras.utils.plot_model(model, to_file="SBERT_predict.jpg", show_shapes=True, dpi=900)

        model.summary(line_length=200)
        for tv in model.variables:
            print(tv.name, tv.shape)

        return model

    def train(self):
        model = self.build_model()
        if PARAMS_mode == 'train0':
            load_model_weights_from_checkpoint_bert(model, PARAMS_bert_path + "/bert_model.ckpt")

            decay_schedule = PolynomialDecay(initial_learning_rate=PARAMS_lr,
                                             decay_steps=PARAMS_decay_steps,
                                             end_learning_rate=0.0,
                                             power=1.0,
                                             cycle=False)

            warmup_schedule = WarmUp(initial_learning_rate=PARAMS_lr,
                                     decay_schedule_fn=decay_schedule,
                                     warmup_steps=PARAMS_warmup_steps,
                                     )

            optimizer = AdamWeightDecay(learning_rate=warmup_schedule,
                                        weight_decay_rate=0.01,
                                        epsilon=1.0e-6,
                                        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
        else:
            model.load_weights(PARAMS_check + '/SBERT.h5')

            optimizer = AdamWeightDecay(learning_rate=PARAMS_lr,
                                        weight_decay_rate=0.01,
                                        epsilon=1.0e-6,
                                        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

        lossobj = BinaryCrossentropy()
        model.compile(optimizer=optimizer, loss=lossobj, metrics=["acc"], )
        model.save(PARAMS_check + '/SBERT.h5')

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=PARAMS_patience),
            ModelCheckpoint(filepath=PARAMS_check + '/SBERT.h5',
                            monitor='val_loss',
                            save_best_only=True)
        ]

        train_batch = batched_data(PARAMS_train_file,
                                   single_example_parser_SBERT,
                                   PARAMS_batch_size,
                                   padded_shapes=(([-1], [-1]), []))
        dev_batch = batched_data(PARAMS_dev_file,
                                 single_example_parser_SBERT,
                                 PARAMS_batch_size,
                                 padded_shapes=(([-1], [-1]), []))

        history = model.fit(
            train_batch,
            epochs=PARAMS_epochs,
            validation_data=dev_batch,
            callbacks=callbacks
        )

        file = open(PARAMS_check + '/history.pkl', 'wb')
        pickle.dump(history.history, file)
        file.close()

    def predict(self, sentencesa, sentencesb):
        sen2ida = self.tokenizer(sentencesa, padding=True, return_tensors="tf")["input_ids"]
        sen2idb = self.tokenizer(sentencesb, padding=True, return_tensors="tf")["input_ids"]

        model = self.build_model()
        model.load_weights(PARAMS_check + "/SBERT.h5", by_name=True)

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
        ],
            [
                "怎么再二次修改微信号",
                "怎么才能赚钱",
                "我不喜欢北京",
                "北京是中国的首都",
            ])
