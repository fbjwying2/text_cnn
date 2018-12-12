# -*- coding: utf-8 -*-
import codecs
import random
import numpy as np
from tflearn.data_utils import pad_sequences
from collections import Counter
import os
import pickle

PAD_ID = 0
UNK_ID=1
_PAD="_PAD"
_UNK="UNK"


def load_data_multilabel(traning_data_path,vocab_word2index, vocab_label2index,sentence_len,training_portion=0.95):
    """
    convert data as indexes using word2index dicts.
    :param traning_data_path:
    :param vocab_word2index:
    :param vocab_label2index:
    :return:
    """
    file_object = codecs.open(traning_data_path, mode='r', encoding='utf-8')
    lines = file_object.readlines()
    random.shuffle(lines)
    label_size=len(vocab_label2index)
    X = []
    Y = []
    for i,line in enumerate(lines):
        raw_list = line.strip().split("__label__")
        input_list = raw_list[0].strip().split(" ")
        input_list = [x.strip().replace(" ", "") for x in input_list if x != '']
        x=[vocab_word2index.get(x,UNK_ID) for x in input_list]
        label_list = raw_list[1:]
        label_list=[l.strip().replace(" ", "") for l in label_list if l != '']
        label_list=[vocab_label2index[label] for label in label_list]
        y=transform_multilabel_as_multihot(label_list,label_size)
        X.append(x)
        Y.append(y)
        if i<10:print(i,"line:",line)

    X = pad_sequences(X, maxlen=sentence_len, value=0.)  # padding to max length
    number_examples = len(lines)
    training_number=int(training_portion* number_examples)
    train = (X[0:training_number], Y[0:training_number])

    test_number=int((number_examples-training_number)/2)


    test = (X[training_number+ 1:training_number+test_number], Y[training_number + 1:training_number+test_number])
    valid = (X[training_number + test_number + 1:],
             Y[training_number + test_number + 1:])

    return train,test,valid


def transform_multilabel_as_multihot(label_list,label_size):
    """
    convert to multi-hot style
    :param label_list: e.g.[0,1,4], here 4 means in the 4th position it is true value(as indicate by'1')
    :param label_size: e.g.199
    :return:e.g.[1,1,0,1,0,0,........]
    """
    result=np.zeros(label_size)
    #set those location as 1, all else place as 0.
    result[label_list] = 1
    return result

#use pretrained word embedding to get word vocabulary and labels, and its relationship with index
def create_vocabulary(training_data_path,vocab_size,name_scope='cnn'):
    """
    create vocabulary
    :param training_data_path:
    :param vocab_size:
    :param name_scope:
    :return:
    """

    cache_vocabulary_label_pik='cache'+"_"+name_scope # path to save cache
    if not os.path.isdir('../output/' + cache_vocabulary_label_pik): # create folder if not exists.
        os.makedirs(cache_vocabulary_label_pik)

    # if cache exists. load it; otherwise create it.
    cache_path ='../output/' + cache_vocabulary_label_pik+"/"+'vocab_label.pik'
    print("cache_path:",cache_path,"file_exists:",os.path.exists(cache_path))
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as data_f:
            return pickle.load(data_f)
    else:
        vocabulary_word2index={}
        vocabulary_index2word={}
        vocabulary_word2index[_PAD]=PAD_ID
        vocabulary_index2word[PAD_ID]=_PAD
        vocabulary_word2index[_UNK]=UNK_ID
        vocabulary_index2word[UNK_ID]=_UNK

        vocabulary_label2index={}
        vocabulary_index2label={}

        #1.load raw data
        file_object = codecs.open(training_data_path, mode='r', encoding='utf-8')
        lines=file_object.readlines()
        #2.loop each line,put to counter
        c_inputs=Counter()
        c_labels=Counter()
        l_lbs = {}
        for i, line in enumerate(lines):
            raw_list=line.strip().split("__label__")

            input_list = raw_list[0].strip().split(" ")
            input_list = [x.strip().replace(" ", "") for x in input_list if x != '']
            label_list=[l.strip().replace(" ","") for l in raw_list[1:] if l!='']
            c_inputs.update(input_list)
            c_labels.update(label_list)

            if i > 4427:
                k = 1

            if len(label_list) > 0 and label_list[0] not in l_lbs:
                l_lbs[label_list[0]] = 1

        #return most frequency words
        vocab_list=c_inputs.most_common(vocab_size)
        label_list=c_labels.most_common()
        #put those words to dict
        for i,tuplee in enumerate(vocab_list):
            word,_=tuplee
            vocabulary_word2index[word]=i+2
            vocabulary_index2word[i+2]=word

        for i,tuplee in enumerate(label_list):
            label,_=tuplee;label=str(label)
            vocabulary_label2index[label]=i
            vocabulary_index2label[i]=label

        #save to file system if vocabulary of words not exists.
        if not os.path.exists(cache_path):
            with open(cache_path, 'ab') as data_f:
                pickle.dump((vocabulary_word2index,vocabulary_index2word,vocabulary_label2index,vocabulary_index2label), data_f)
    return vocabulary_word2index,vocabulary_index2word,vocabulary_label2index,vocabulary_index2label


def sort_by_value(d):
    items = d.items()
    backitems = [[v[1], v[0]] for v in items]
    backitems.sort(reverse=True)
    return [backitems[i][1] for i in range(0, len(backitems))]


# create vocabulary of lables. label is sorted. 1 is high frequency, 2 is low frequency.
def create_voabulary_label(voabulary_label, name_scope='', use_seq2seq=False):
    print("create_voabulary_label_sorted.started.traning_data_path:", voabulary_label)
    cache_path = '../output/cache_vocabulary_label_pik/' + name_scope + "_label_voabulary.pik"
    if os.path.exists(cache_path):  # 如果缓存文件存在，则直接读取
        with open(cache_path, 'rb') as data_f:
            vocabulary_word2index_label, vocabulary_index2word_label, = pickle.load(data_f)
            return vocabulary_word2index_label, vocabulary_index2word_label
    else:
        zhihu_f_train = codecs.open(voabulary_label, 'r', encoding='utf-8')
        lines = zhihu_f_train.readlines()
        count = 0
        vocabulary_word2index_label = {}
        vocabulary_index2word_label = {}
        vocabulary_label_count_dict = {}  # {label:count}
        for i, line in enumerate(lines):
            if '__label__' in line:  # '__label__-2051131023989903826
                label = line[line.index('__label__') + len('__label__'):].strip().replace("\n", "")
                if vocabulary_label_count_dict.get(label, None) is not None:
                    vocabulary_label_count_dict[label] = vocabulary_label_count_dict[label] + 1
                else:
                    vocabulary_label_count_dict[label] = 1
        list_label = sort_by_value(vocabulary_label_count_dict)

        print("length of list_label:", len(list_label));  # print(";list_label:",list_label)
        countt = 0

        ##########################################################################################
        if use_seq2seq:  # if used for seq2seq model,insert two special label(token):_GO AND _END
            i_list = [0, 1, 2];
            label_special_list = [_GO, _END, _PAD]
            for i, label in zip(i_list, label_special_list):
                vocabulary_word2index_label[label] = i
                vocabulary_index2word_label[i] = label
        #########################################################################################

        for i, label in enumerate(list_label):
            if i < 10:
                count_value = vocabulary_label_count_dict[label]
                print("label:", label, "count_value:", count_value)
                countt = countt + count_value

            indexx = i + 3 if use_seq2seq else i
            vocabulary_word2index_label[label] = indexx
            vocabulary_index2word_label[indexx] = label

        print("count top10:", countt)

        # save to file system if vocabulary of words is not exists.
        if not os.path.exists(cache_path):  # 如果不存在写到缓存文件中
            with open(cache_path, 'ab') as data_f:
                pickle.dump((vocabulary_word2index_label, vocabulary_index2word_label), data_f)

    print("create_voabulary_label_sorted.ended.len of vocabulary_label:", len(vocabulary_index2word_label))
    return vocabulary_word2index_label, vocabulary_index2word_label

def load_data(cache_file_h5py,cache_file_pickle):
    """
    load data from h5py and pickle cache files, which is generate by take step by step of pre-processing.ipynb
    :param cache_file_h5py:
    :param cache_file_pickle:
    :return:
    """
    if not os.path.exists(cache_file_h5py) or not os.path.exists(cache_file_pickle):
        raise RuntimeError("############################ERROR##############################\n. "
                           "please download cache file, it include training data and vocabulary & labels. "
                           "link can be found in README.md\n download zip file, unzip it, then put cache files as FLAGS."
                           "cache_file_h5py and FLAGS.cache_file_pickle suggested location.")
    print("INFO. cache file exists. going to load cache file")
    f_data = h5py.File(cache_file_h5py, 'r')
    print("f_data.keys:",list(f_data.keys()))
    train_X=f_data['train_X'] # np.array(
    print("train_X.shape:",train_X.shape)
    train_Y=f_data['train_Y'] # np.array(
    print("train_Y.shape:",train_Y.shape,";")
    vaild_X=f_data['vaild_X'] # np.array(
    valid_Y=f_data['valid_Y'] # np.array(
    test_X=f_data['test_X'] # np.array(
    test_Y=f_data['test_Y'] # np.array(
    #print(train_X)
    #f_data.close()

    word2index, label2index=None,None
    with open(cache_file_pickle, 'rb') as data_f_pickle:
        word2index, label2index=pickle.load(data_f_pickle)
    print("INFO. cache file load successful...")
    return word2index, label2index,train_X,train_Y,vaild_X,valid_Y,test_X,test_Y

#training_data_path='../data/sample_multiple_label3.txt'
#vocab_size=100
#create_voabulary(training_data_path,vocab_size)
