'''
    基于SentenceBERT的中文相似度模型
    简化bert调用
'''
import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from official.nlp.optimization import WarmUp, AdamWeightDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from transformers import BertTokenizer, TFBertModel, BertConfig
from math import ceil
import pickle

import os

from FuncUtils import get_cos_distance

from TFDataUtils import single_example_parser_SBERT, batched_data

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

PARAMS_model = "SBERT_lcqmc_simplified"
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

        Config = BertConfig.from_pretrained("hfl/chinese-roberta-wwm-ext")
        Config.attention_probs_dropout_prob = PARAMS_drop_rate
        Config.hidden_dropout_prob = PARAMS_drop_rate

        self.bert = TFBertModel.from_pretrained("hfl/chinese-roberta-wwm-ext", config=Config)

    def call(self, inputs, **kwargs):
        return self.bert(inputs)[0]


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

        now = BERT(name="bert")(now)

        now = GlobalAveragePool(name="globalaveragepool")(now)

        if PARAMS_mode.startswith("train"):
            logits = ProjectTrain(name="projecttrain")(now)
            model = Model(inputs=[sena, senb], outputs=logits)
            # tf.keras.utils.plot_model(model, to_file="SBERT_train_simplified.jpg", show_shapes=True, dpi=900)
        else:
            logits = ProjectPredict(name="projectpredict")(now)
            model = Model(inputs=[sena, senb], outputs=logits)
            # tf.keras.utils.plot_model(model, to_file="SBERT_predict_simplified.jpg", show_shapes=True, dpi=900)

        model.summary(line_length=200)
        for tv in model.variables:
            print(tv.name, tv.shape)

        return model

    def train(self):
        model = self.build_model()
        if PARAMS_mode == 'train0':
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
