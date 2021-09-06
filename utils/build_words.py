import json
import os
import numpy as np
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import tqdm


def get_labels(label_file):
    with open(label_file, 'r') as fr:
        id2label_, label2id_ = json.load(fr)
    id2label, label2id = {}, {}
    for key, value in id2label_.items():
        id2label[int(key)] = str(value)
    for key, value in label2id_.items():
        label2id[str(key)] = int(value)
    return id2label, label2id


def convert_glove_format(w2v_path, glove_out_file):
    if not os.path.exists(glove_out_file):
        glove2word2vec(w2v_path, glove_out_file)


def load_glove(glove_out_file):
    print("start to load glove")
    word2vec = KeyedVectors.load_word2vec_format(glove_out_file, binary=False)
    print("end to load glove")
    return word2vec


def build_words(word2vec, output_dir, vocab_dir, do_lower_case=False, add_begin_token=False):
    with open("{}/{}_glove_word_list.txt".format(vocab_dir, 'cased' if do_lower_case else "uncased"), 'w') as fw:
        for vocab in word2vec.vocab:
            fw.write(vocab+'\n')

    type_list = ['Positive', 'Negative']
    output_prefix = 'emotional_orientation'
    output_prefix += '_uncased' if do_lower_case else "_cased"
    output_prefix += '_with_begin' if add_begin_token else ""
    wordset = set()
    event_str_ids, event_type_word_list, event_type_idx, start = [], [], [], 0

    label2id, id2label = {}, {}

    hit, dim = 0, 0
    uwk_vec = np.random.uniform(low=-1.0, high=1.0, size=dim)
    vecs = []

    for idx, label_type_name in enumerate(type_list):
        idx_ = idx+1 if add_begin_token else idx
        label2id[label_type_name] = idx_
        id2label[idx_] = label_type_name
        cur_words = label_type_name.split('-')
        if do_lower_case:
            cur_words = [word.lower() for word in cur_words]
        cur_vecs = []
        exist = True
        for word in cur_words:
            word_ = word.lower()
            if word in word2vec:
                dim = len(word2vec[word])
                cur_vecs.append(word2vec[word])
            else:
                exist = False
                cur_vecs.append(uwk_vec)
        if not exist:
            print("{} doesn't exist".format(label_type_name))
        else:
            hit += 1
        cur_vecs = np.array(cur_vecs)
        vecs.append(np.average(cur_vecs, axis=0))

    assert len(vecs) == len(label2id) == len(id2label) == len(type_list)

    event_str_ids = np.array(
        [idx for idx in range(len(label2id))], dtype=np.int)
    print("total {} label; vector dim: {}; hit in glove:{}/{}".format(len(label2id),
                                                                      dim, hit, len(label2id)))
    np.save("{}/glove_{}_whole_{}d.npy".format(output_dir, output_prefix, dim),
            np.array(vecs, dtype=np.float32))

    # with open('{}/{}_whole_word_map.json'.format(output_dir, output_prefix), 'w', encoding='utf-8') as fw:
    #     json.dump([id2label, label2id], fw, indent=4, ensure_ascii=False)

    np.save("{}/{}_whole_ids.npy".format(output_dir,
                                         output_prefix), event_str_ids)


def check_label():
    word_dict = {}
    with open('/home/yangpan/workspace/pretrain_models/cased_glove_word_list.txt', 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            word = line.strip()
            word_dict[word] = len(word_dict)

    for prefix in ['trigger', 'argument']:
        hit = 0
        print("####\n{}".format(prefix))
        type_list = get_labels(
            '/home/yangpan/workspace/onepass_ie/data/ace05/splited/{}_label_map.json'.format(prefix))[1].keys()
        for item in type_list:
            cur_words = item.split('-')
            exist = True
            for word in cur_words:
                word_ = word.lower()
                if word_ not in word_dict:
                    exist = False
            if not exist:
                print("{} : {}".format(item, word_))
            else:
                hit += 1
        print("hit {}/{}".format(hit, len(type_list)))


if __name__ == "__main__":
    w2v_path = '/home/yangpan/workspace/pretrain_models/glove.840B.300d.txt'
    glove_out_file = '/home/yangpan/workspace/pretrain_models/glove.840B.300d_word2vec.txt'
    output_dir = '/home/yangpan/workspace/dataset/imdb'
    convert_glove_format(w2v_path, glove_out_file)
    word2vec = load_glove(glove_out_file)
    build_words(word2vec, output_dir,
                '/home/yangpan/workspace/pretrain_models/', add_begin_token=False)

    # w2v_path = '/home/yangpan/workspace/pretrain_models/glove.42B.300d.txt'
    # glove_out_file = '/home/yangpan/workspace/pretrain_models/glove.42B.300d_word2vec.txt'
    # convert_glove_format(w2v_path, glove_out_file)
    # word2vec = load_glove(glove_out_file)
    # build_words(word2vec, output_dir,
    #             '/home/yangpan/workspace/pretrain_models/', do_lower_case=True, add_begin_token=False)
