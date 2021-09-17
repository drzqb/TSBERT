import tensorflow as tf
from transformers import BertTokenizer
import numpy as np
import jsonlines


def snli2txt(jsonlfile, txtfile, tfrecordfile):
    sen1 = dict()
    fw = open(txtfile, "w", encoding="utf-8")

    with jsonlines.open(jsonlfile) as reader:
        for obj in reader:
            s1 = obj["sentence1"]
            if s1 in sen1.keys():
                ls = len(sen1[s1].keys())
                if ls < 2:
                    label = obj["gold_label"]
                    if label != "neutral" and label not in sen1[s1].keys():
                        sen1[s1][label] = obj["sentence2"]
            else:
                sen1[s1] = dict()
                label = obj["gold_label"]
                if label != "neutral":
                    sen1[s1][label] = obj["sentence2"]

    print("samples all: ", len(sen1.keys()))

    msamples = 0
    for k, v in sen1.items():
        if len(v.keys()) == 2:
            fw.write(k + "\t" + v["entailment"] + "\t" + v["contradiction"] + "\n")
            msamples += 1
    fw.close()

    print("samples succeed: ", msamples)

    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    writer = tf.io.TFRecordWriter(tfrecordfile)
    num_example = 0
    for k, v in sen1.items():
        if len(v.keys()) == 2:
            if np.random.random() > 0.1:
                continue
            else:
                sen_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in
                               tokenizer(k)["input_ids"]]
                senplus_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in
                                   tokenizer(v["entailment"])["input_ids"]]
                senminus_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in
                                    tokenizer(v["contradiction"])["input_ids"]]
                seq_example = tf.train.SequenceExample(
                    feature_lists=tf.train.FeatureLists(feature_list={
                        'sen': tf.train.FeatureList(feature=sen_feature),
                        'senplus': tf.train.FeatureList(feature=senplus_feature),
                        'senminus': tf.train.FeatureList(feature=senminus_feature),
                    }),
                )

                serialized = seq_example.SerializeToString()
                writer.write(serialized)
                num_example += 1

                print("\rnum_example: %d" % (num_example), end="")

                if num_example == 10000:
                    break


def single_example_parser_SimCSE_snli(serialized_example):
    sequence_features = {
        'sen': tf.io.FixedLenSequenceFeature([], tf.int64),
        'senplus': tf.io.FixedLenSequenceFeature([], tf.int64),
        'senminus': tf.io.FixedLenSequenceFeature([], tf.int64),
    }

    _, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=serialized_example,
        sequence_features=sequence_features
    )

    sen = sequence_parsed['sen']
    senplus = sequence_parsed['senplus']
    senminus = sequence_parsed['senminus']

    return (sen, senplus, senminus), tf.constant(1, dtype=tf.int64)


def sentence2tfrecord_SBERT(tsvfile, tfrecordfile):
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    writer = tf.io.TFRecordWriter(tfrecordfile)
    num_example = 0

    with  open(tsvfile, 'rt', encoding='utf-8') as f:
        next(f)

        for line in f:
            lines = line.lower().split("\t")
            sena, senb, label = lines[0].strip(), lines[1].strip(), int(lines[2].strip())

            sen2ida = tokenizer(sena)["input_ids"]
            sen2idb = tokenizer(senb)["input_ids"]

            sena_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in
                            sen2ida]
            senb_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in
                            sen2idb]
            label_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))

            seq_example = tf.train.SequenceExample(
                feature_lists=tf.train.FeatureLists(feature_list={
                    'sena': tf.train.FeatureList(feature=sena_feature),
                    'senb': tf.train.FeatureList(feature=senb_feature),
                }),
                context=tf.train.Features(feature={
                    'label': label_feature
                }),
            )

            serialized = seq_example.SerializeToString()
            writer.write(serialized)
            num_example += 1

            print("\r num_example: %d" % (num_example), end="")  # train: 238766


def single_example_parser_SBERT(serialized_example):
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

    return (sena, senb), label


def sentence2tfrecord_SimCSE(tsvfile, tfrecordfile):
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    writer = tf.io.TFRecordWriter(tfrecordfile)
    num_example = 0

    with  open(tsvfile, 'rt', encoding='utf-8') as f:
        next(f)

        for line in f:
            lines = line.lower().split("\t")
            sena = lines[0].strip()

            sen2id = tokenizer(sena)["input_ids"]

            sen_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in
                           sen2id]

            seq_example = tf.train.SequenceExample(
                feature_lists=tf.train.FeatureLists(feature_list={
                    'sen': tf.train.FeatureList(feature=sen_feature),
                }),
            )

            serialized = seq_example.SerializeToString()
            writer.write(serialized)
            num_example += 1

            print("\r num_example: %d" % (num_example), end="")  # train: 238766


def single_example_parser_SimCSE(serialized_example):
    sequence_features = {
        'sen': tf.io.FixedLenSequenceFeature([], tf.int64),
    }

    _, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=serialized_example,
        sequence_features=sequence_features
    )

    sen = sequence_parsed['sen']
    return sen, tf.constant(1, dtype=tf.int64)


def single_example_parser_SimCSE_endpoint(serialized_example):
    sequence_features = {
        'sen': tf.io.FixedLenSequenceFeature([], tf.int64),
    }

    _, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=serialized_example,
        sequence_features=sequence_features
    )

    sen = sequence_parsed['sen']
    return {"sen": sen}


def single_example_parser_SimCSE_endpoints(serialized_example):
    sequence_features = {
        'sen': tf.io.FixedLenSequenceFeature([], tf.int64),
    }

    _, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=serialized_example,
        sequence_features=sequence_features
    )

    sen = sequence_parsed['sen']
    return {"sena": sen, "senb": sen}


def batched_data(tfrecord_filename, single_example_parser, batch_size, padded_shapes, shuffle=True, repeat=True):
    dataset = tf.data.TFRecordDataset(tfrecord_filename)
    if shuffle:
        dataset = dataset.shuffle(100 * batch_size)

    dataset = dataset.map(single_example_parser) \
        .padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=False) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    if repeat:
        dataset = dataset.repeat()

    return dataset


if __name__ == "__main__":
    # sentence2tfrecord_SimCSE("data/OriginalFiles/lcqmc/lcqmc_train.tsv",
    #                          "data/TFRecordFiles/simcse_train_lcqmc.tfrecord")    # 238766

    # sentence2tfrecord_SimCSE("data/OriginalFiles/lcqmc/lcqmc_dev.tsv",
    #                          "data/TFRecordFiles/simcse_dev_lcqmc.tfrecord")    # 8802

    sentence2tfrecord_SimCSE("data/OriginalFiles/lcqmc/lcqmc_test.tsv",
                             "data/TFRecordFiles/simcse_test_lcqmc.tfrecord")  # 12500

    # snli2txt("data/OriginalFiles/snli/cnsd_snli_v1.0.train.jsonl",
    #          "data/OriginalFiles/snli/cnsd_snli_v1.0.train.txt",
    #          "data/TFRecordFiles/cnsd_snli_v1.0.train.tfrecord",
    #          )

    # sentence2tfrecord_SBERT("data/OriginalFiles/lcqmc/lcqmc_dev.tsv",
    #                         "data/TFRecordFiles/lcqmc_dev.tfrecord")
