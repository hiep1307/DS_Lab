from os import listdir
from os.path import isfile
import re
from collections import defaultdict
import numpy as np
def gather_20newsgroup_data():     
    path = './20news-bydate/'
    dirs = [path + dir_name + '/' for dir_name in listdir(path) if not isfile(path+dir_name)]
    train_dir, test_dir = (dirs[0], dirs[1]) if 'train' in dirs[0] else (dirs[1], dirs[0])
    list_newsgroups = [newsgroup for newsgroup in listdir(train_dir)]
    list_newsgroups.sort()      # name of each data folder 
#   ........
    with open('./20news-bydate/stop_words.txt') as f:
        stop_words = f.read().splitlines()
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    def collect_data_from(parent_dir, newsgroup_list):
        data = []
        for group_id, newsgroup in enumerate(newsgroup_list):
            label = group_id
            dir_path = parent_dir + '/' + newsgroup +'/'
            files = [(filename, dir_path + filename) for filename in listdir(dir_path) if isfile(dir_path + filename)]  # list of (file_name, file_path)
            files.sort()
            for file_name, file_path in files:
                with open(file_path, encoding= 'ISO-8859-15') as f:
                    text = f.read().lower()
                    words = [stemmer.stem(word) for word in re.split('\W+', text) if word not in stop_words]
                    content = ' '.join(words)
                    assert len(content.splitlines()) == 1
                    data.append(str(label) + '<fff>' + file_name + '<fff>' + content)
        return data
    train_data = collect_data_from(parent_dir= train_dir, newsgroup_list= list_newsgroups)
    test_data = collect_data_from(parent_dir= test_dir, newsgroup_list= list_newsgroups)
    full_data = train_data + test_data
    with open('./20news-bydate/20news-train-processed.txt', 'w') as f:
        f.write('\n'.join(train_data))
    with open('./20news-bydate/20news-test-processed.txt', 'w') as f:
        f.write('\n'.join(test_data))
    with open('./20news-bydate/20news-full-processed.txt', 'w') as f:
        f.write('\n'.join(full_data))

def generate_vocabulary(data_path):
    def compute_idf(df, corpus_size):
        assert df > 0
        return np.log10(corpus_size * 1. / df)
    with open(data_path) as f:
        lines = f.read().splitlines()
    doc_count = defaultdict(int)
    corpus_size = len(lines)
    for line in lines:
        features = line.split('<fff>')
        text = features[-1]
        words = list(set(text.split()))   
        for word in words:
            doc_count[word] += 1        #number of doc has the word
    words_idfs = [(word, compute_idf(document_freq, corpus_size)) for word, document_freq in zip(doc_count.keys(), doc_count.values()) if document_freq >10 and not word.isdigit()]
    words_idfs.sort(key= lambda element: -element[1])   #(key= lambda element: -element[1])    
    print('Vocabulary size: {}'.format(len(words_idfs)))
    with open('./20news-bydate/words_idfs.txt', 'w') as f:
        f.write('\n'.join([word + '<fff>' + str(idf) for word, idf in words_idfs]))     #word + idf

def get_tf_idf(data_path, file_path):      #get_tf_idf of each doc and normalize 
    with open('./20news-bydate/words_idfs.txt', 'r') as f:
        lines = f.read().splitlines()
    words_idfs = [(line.split('<fff>')[0], float(line.split('<fff>')[1])) for line in lines]
    word_IDs = dict([(word, index) for index, (word, idf) in enumerate(words_idfs)])
    idfs = dict(words_idfs)
    with open(data_path) as f:     
        documents = [(int(line.split('<fff>')[0]), int(line.split('<fff>')[1]), line.split('<fff>')[2]) for line in f.read().splitlines()]
    data_iftdf = []
    for document in documents:
        label, doc_id, text = document
        words = [word for word in text.split() if word in idfs]
        word_set = set(words)
        max_word_freq = max([words.count(word) for word in word_set])
        words_tfidf = []        #list of (word_ID, word_tfidf)
        sum_squares = 0.0
        # compute tf_idf_normalized of each word
        for word in word_set:
            word_freq = words.count(word)
            word_tfidf = word_freq * 1. / max_word_freq * idfs[word]
            words_tfidf.append((word_IDs[word], word_tfidf))        
            sum_squares += word_tfidf ** 2
        words_tfidf_normalized = [str(index) + ':' + str(tfidf_value * 1. / np.sqrt(sum_squares)) for index, tfidf_value in words_tfidf]
        sparse_rep = ' '.join(words_tfidf_normalized)
        data_iftdf.append((label, doc_id, sparse_rep))
    with open(file_path, 'w') as f:
        f.write('\n'.join([str(label) + '<fff>' + str(doc_id) + '<fff>' + sparse_rep for label, doc_id, sparse_rep in data_iftdf]))
# gather_20newsgroup_data()
# generate_vocabulary('./20news-bydate/20news-train-processed.txt')
get_tf_idf('./20news-bydate/20news-train-processed.txt', './20news-bydate/20news-train-tfidf.txt')
get_tf_idf('./20news-bydate/20news-test-processed.txt', './20news-bydate/20news-test-tfidf.txt')