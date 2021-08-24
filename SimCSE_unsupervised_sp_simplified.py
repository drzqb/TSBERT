'''
    基于SimCSE无监督的中文相似度模型
    采用苏神的loss
    简化bert调用
'''
import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from transformers import TFBertModel, BertTokenizer, BertConfig
import pickle

import os

from FuncUtils import get_cos_distance

from TFDataUtils import single_example_parser_SimCSE, batched_data

PARAMS_hidden_size = 768
PARAMS_batch_size = 16

PARAMS_mode = "predict"
PARAMS_epochs = 1
PARAMS_lr = 1.0e-5

PARAMS_train_file = [
    'data/TFRecordFiles/simcse_train_lcqmc.tfrecord',
]

PARAMS_dev_file = [
    'data/TFRecordFiles/lcqmc_test.tfrecord',
]

PARAMS_model = "SimCSE_lcqmc_sp_simplified"
PARAMS_check = "modelfiles/" + PARAMS_model

PARAMS_drop_rate = 0.3


class Aug(Layer):
    def __init__(self, **kwargs):
        super(Aug, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        if isinstance(inputs, tuple):  # predict stage
            x, y = inputs
            batchsize = tf.shape(x)[0]
            lx = tf.shape(x)[1]
            ly = tf.shape(y)[1]
            lmax = tf.maximum(lx, ly)
            xx = tf.concat([x, tf.zeros([batchsize, lmax - lx], tf.int32)], axis=-1)
            yy = tf.concat([y, tf.zeros([batchsize, lmax - ly], tf.int32)], axis=-1)

            return tf.concat([xx, yy], axis=0)
        else:  # train stage
            batchsize = tf.shape(inputs)[0]
            res = tf.tile(inputs, [1, 2])

            return tf.reshape(res, [batchsize * 2, -1])


class BERT(Layer):
    def __init__(self, **kwargs):
        super(BERT, self).__init__(**kwargs)

        Config = BertConfig.from_pretrained("hfl/chinese-roberta-wwm-ext")
        Config.attention_probs_dropout_prob = PARAMS_drop_rate
        Config.hidden_dropout_prob = PARAMS_drop_rate

        self.bert = TFBertModel.from_pretrained("hfl/chinese-roberta-wwm-ext", config=Config)

    def call(self, inputs, **kwargs):
        return self.bert(inputs)[0]


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


class ProjectPredict(Layer):
    def __init__(self, **kwargs):
        super(ProjectPredict, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        u, v = tf.split(inputs, 2)
        cos = get_cos_distance(u, v)
        similarity = (1.0 + cos) / 2.0

        return similarity


def CustomLoss(y_true, y_pred):
    idxs = tf.range(0, tf.shape(y_pred)[0])
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    y_true = tf.equal(idxs_1, idxs_2)
    y_true = tf.cast(y_true, tf.float32)
    # 计算相似度
    y_pred = tf.nn.l2_normalize(y_pred, axis=1)
    similarities = tf.matmul(y_pred, y_pred, transpose_b=True)
    similarities = similarities - tf.eye(tf.shape(y_pred)[0]) * 1e12
    similarities = similarities * 20
    loss = categorical_crossentropy(y_true, similarities, from_logits=True)
    return tf.reduce_mean(loss)


class USER():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    def build_model_train(self):
        sen = Input(shape=[None], name='sen', dtype=tf.int32)

        now = Aug(name="aug")(sen)

        now = BERT(name="bert")(now)

        logits = CLSPool(name="clspool")(now)

        model = Model(inputs=sen, outputs=logits)

        # tf.keras.utils.plot_model(model, to_file="SimCSE_train_unsupervised_sp_simplified.jpg", show_shapes=True,
        #                           dpi=900)

        model.summary(line_length=200)
        for tv in model.variables:
            print(tv.name, tv.shape)

        return model

    def build_model_predict(self):
        sena = Input(shape=[None], name='sena', dtype=tf.int32)
        senb = Input(shape=[None], name='senb', dtype=tf.int32)

        now = Aug(name="aug")(inputs=(sena, senb))

        now = BERT(name="bert")(now)

        now = CLSPool(name="clspool")(now)

        logits = ProjectPredict(name="projectpredict")(now)

        model = Model(inputs=[sena, senb], outputs=logits)
        # tf.keras.utils.plot_model(model, to_file="SimCSE_predict_unsupervised_sp_simplified.jpg", show_shapes=True,
        #                           dpi=900)

        model.summary(line_length=200)
        for tv in model.variables:
            print(tv.name, tv.shape)

        return model

    def train(self):
        model = self.build_model_train()

        optimizer = Adam(PARAMS_lr)
        model.compile(optimizer=optimizer,
                      loss=CustomLoss)
        model.save(PARAMS_check + '/SimCSE.h5')

        train_batch = batched_data(PARAMS_train_file,
                                   single_example_parser_SimCSE,
                                   PARAMS_batch_size,
                                   padded_shapes=([-1], []))
        history = model.fit(
            train_batch,
            epochs=PARAMS_epochs,
        )
        model.save(PARAMS_check + "/SimCSE.h5")

        file = open(PARAMS_check + '/history.pkl', 'wb')
        pickle.dump(history.history, file)
        file.close()

    def predict(self, sentencesa, sentencesb):
        sen2ida = self.tokenizer(sentencesa, padding=True, return_tensors="tf")["input_ids"]
        sen2idb = self.tokenizer(sentencesb, padding=True, return_tensors="tf")["input_ids"]

        model = self.build_model_predict()
        model.load_weights(PARAMS_check + "/SimCSE.h5", by_name=True)

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
