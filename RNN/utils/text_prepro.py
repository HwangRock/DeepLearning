import re
import collections
import torch
import unicodedata

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def load_nmt_data(source, target, file_path):
    english = file_path + '.' + source
    deutsche = file_path + '.' + target

    eng_text = []
    deu_text = []

    lines = open(english, encoding='utf-8').read().strip().split('\n')
    for l in lines:
        eng_text.append(clean_str(l, True))

    lines = open(deutsche, encoding='utf-8').read().strip().split('\n')
    for l in lines:
        deu_text.append(clean_str(l, False))

    return eng_text, deu_text



def load_nmt_pair_data(file_path):
    print("Reading lines...")
    # 파일 읽고 줄 단위로 나누기
    lines = open(file_path, encoding='utf-8').read().strip().split('\n')
    # 각 줄을 pairs 단위로 나누고 normalize 함수 진행
    source_text = []
    target_text = []
    for l in lines:
        s, t = l.split('\t')
        source_text.append(clean_str(s, True))
        target_text.append(clean_str(t, True))

    return source_text, target_text

def clean_str(string, nmt=False):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    if nmt:
        return unicodeToAscii(string.strip().lower())
    else:
        return string.strip().lower()

def load_snips_data(file_path, label_dictionary):
    # Load data from files
    text = list(open(file_path+"/seq.in", "r", encoding='UTF-8').readlines())
    text = [clean_str(sent) for sent in text]
    labels_text = list(open(file_path+"/label", "r", encoding='UTF-8').readlines())
    labels_text = [label.strip() for label in labels_text]

    if len(label_dictionary) == 0:
        label_set = set(labels_text)
        for i, label in enumerate(label_set):
            label_dictionary[label] = i
    labels = [label_dictionary[label_text] for label_text in labels_text]
    return text, labels, label_dictionary

def load_mr_data(pos_file, neg_file):
    pos_text = list(open(pos_file, "r", encoding='latin-1').readlines()) # 긍정적인 review 읽어서 list 형태로 관리
    pos_text = [clean_str(sent) for sent in pos_text] # clean_str 함수로 전처리 (소문자, 특수 기호 제거, (), 등 분리)

    neg_text = list(open(neg_file, "r", encoding='latin-1').readlines()) # 부정적인 review 읽어서 list 형태로 관리
    neg_text = [clean_str(sent) for sent in neg_text]

    positive_labels = [1 for _ in pos_text] # 긍정 review 개수만큼 ground_truth 생성
    negative_labels = [0 for _ in neg_text] # 부정 review 개수만큼 ground_truth 생성
    y = positive_labels + negative_labels

    x_final = pos_text + neg_text
    return [x_final, y]

def buildVocab(sentences, vocab_size):
    # Build vocabulary
    words = []
    for sentence in sentences:
        words.extend(sentence.split()) # i, am, a, boy, you, are, a, girl
    print("The number of words: ", len(words))
    word_counts = collections.Counter(words)
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_size)]
    # vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)} # a: 0, i: 1...
    return [vocabulary, vocabulary_inv]

def text_to_indices(x_text, word_id_dict, use_unk=False):
    text_indices = []

    for text in x_text:
        words = text.split()
        ids = [2]  # <s>
        for word in words: # i, am, a, boy
            if word in word_id_dict:
                word_id = word_id_dict[word]
            else:  # oov
                if use_unk:
                    word_id = 1 # OOV (out-of-vocabulary)
                else:
                    word_id = len(word_id_dict)
                    word_id_dict[word] = word_id
            ids.append(word_id) # 5, 8, 6, 19
        ids.append(3)  # </s>dd
        text_indices.append(ids)
    return text_indices

def sequence_to_tensor(sequence_list, nb_paddings=(0, 0)):
    nb_front_pad, nb_back_pad = nb_paddings

    max_length = len(max(sequence_list, key=len)) + nb_front_pad + nb_back_pad
    sequence_tensor = torch.LongTensor(len(sequence_list), max_length).zero_()  # 0: <pad>
    print("\n max length: " + str(max_length))
    for i, sequence in enumerate(sequence_list):
        sequence_tensor[i, nb_front_pad:len(sequence) + nb_front_pad] = torch.tensor(sequence)
    return sequence_tensor