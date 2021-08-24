'''
    基于BERTWhitening无监督的中文相似度模型
    简化bert调用
'''
import tensorflow as tf
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.models import Model
from transformers import TFBertModel, BertTokenizer

import os, pickle

from TFDataUtils import single_example_parser_SimCSE_endpoint, batched_data

PARAMS_hidden_size = 768
PARAMS_batch_size = 16
PARAMS_reduct_dim = 256
PARAMS_mode = "predict"

PARAMS_train_file = [
    'data/TFRecordFiles/simcse_train_lcqmc.tfrecord',
]

PARAMS_model = "bertwhitening"
PARAMS_check = "modelfiles/" + PARAMS_model


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


class BERT(Layer):
    def __init__(self, **kwargs):
        super(BERT, self).__init__(**kwargs)

        self.bert = TFBertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

    def call(self, inputs, **kwargs):
        return self.bert(inputs)[0]


class CLSPool(Layer):
    def __init__(self, **kwargs):
        super(CLSPool, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return inputs[:, 0]


class Whiten(Layer):
    def __init__(self, **kwargs):
        super(Whiten, self).__init__(**kwargs)

        with open(PARAMS_check + "/whiten.txt", "rb") as fr:
            whiten = pickle.load(fr)

        self.W = whiten["W"]
        self.mu = whiten["mu"]

    def call(self, inputs, **kwargs):
        z = tf.matmul(inputs - self.mu, self.W)
        z = tf.nn.l2_normalize(z, axis=-1)

        x, y = tf.split(z, 2)
        return (1.0 + tf.reduce_sum(x * y, axis=-1)) / 2.0


class USER():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    def build_model_train(self):
        sen = Input(shape=[None], name='sen', dtype=tf.int32)

        now = BERT(name="bert")(sen)

        logits = CLSPool(name="clspool")(now)

        model = Model(inputs=sen, outputs=logits)

        model.summary(line_length=200)

        return model

    def build_model_predict(self):
        sena = Input(shape=[None], name='sena', dtype=tf.int32)
        senb = Input(shape=[None], name='senb', dtype=tf.int32)

        sen = Aug(name="aug")(inputs=(sena, senb))

        now = BERT(name="bert")(sen)

        logits = CLSPool(name="clspool")(now)

        cos = Whiten(name="whiten")(logits)

        model = Model(inputs=[sena, senb], outputs=cos)

        model.summary(line_length=200)

        return model

    def train(self):
        model = self.build_model_train()

        train_batch = batched_data(PARAMS_train_file,
                                   single_example_parser_SimCSE_endpoint,
                                   PARAMS_batch_size,
                                   padded_shapes={"sen": [-1]})
        mu = tf.zeros(PARAMS_hidden_size)
        sigma = tf.zeros([PARAMS_hidden_size, PARAMS_hidden_size])

        n = 0.0

        for data in train_batch:
            output = model.predict(data)
            for i in range(len(output)):
                mu = (n * mu + output[i]) / (n + 1.0)
                sigma = (n * sigma + tf.matmul(output[i:i + 1] - mu, output[i:i + 1] - mu, transpose_a=True)) / (
                        n + 1.0)
                n += 1.0

        GAMA, U, _ = tf.linalg.svd(sigma)

        W = tf.matmul(U, tf.linalg.diag(1.0 / tf.sqrt(GAMA)))[:, :PARAMS_reduct_dim]

        whiten = {
            'W': W,
            'mu': mu
        }

        with open(PARAMS_check + "/whiten.txt", "wb") as fw:
            pickle.dump(whiten, fw)

    def predict(self, sentencesa, sentencesb):
        sen2ida = self.tokenizer(sentencesa, padding=True, return_tensors="tf")["input_ids"]
        sen2idb = self.tokenizer(sentencesb, padding=True, return_tensors="tf")["input_ids"]

        model = self.build_model_predict()

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
