def load_data(data_path):
    #load data and convert to sparse matrix
    def sparse_to_dense(sparse_r_d, vocab_size):
        r_ds = [0.0 for _ in range(vocab_size)]
        indices_tfidf = sparse_r_d.split()
        for index_tfidf in indices_tfidf:
            id = int(index_tfidf.split(':')[0])
            tfidf = float(index_tfidf.split(':')[1])
            r_ds[id] = tfidf
        return r_ds
    with open(data_path, 'r') as f:
        d_lines = f.read().splitlines()
    with open('20news-bydate/words_idfs.txt', 'r') as f:
        vocab_size = len(f.read().splitlines())
    labels = []
    data = []
    
    for d_line in d_lines:
        label, id_doc, feature = d_line.split('<fff>')
        labels.append(int(label))
        r_d = sparse_to_dense(feature, vocab_size)
        data.append(r_d)
    return data, labels

def clustering_with_Kmeans():
    #using Scikit-learn to cluster with K-means
    data, labels = load_data(data_path= '20news-bydate/20news-full-tfidf.txt')
    # use csr_matrix to create a sparse matrix with efficient row slicing
    from sklearn.cluster import KMeans
    from scipy.sparse import csr_matrix
    X = csr_matrix(data)
    print('========')
    kmeans = KMeans(n_clusters = 20, init = 'random', n_init = 5, tol = 1e-3, random_state= 2018).fit(X)
    labels = kmeans.label_

clustering_with_Kmeans()
