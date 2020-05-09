def load_data(data_path):
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

def compute_accuracy(predicted_y, expected_y):
    #compute accuracy
    matches = np.equal(predicted_y, expected_y)
    accuracy = np.sum(matches.astype(float) / len(expected_y))
    return accuracy

def classifying_with_linear_SVMs():
    #using Scikit-learn to classify with linear SVMs
    train_X, train_y = load_data(data_path= '20news-bydate/20news-train-tfidf.txt')
    from sklearn.svm import LinearSVC
    classifier = LinearSVC(C = 10.0, tol = 0.001, verbose = True)   #penalty coeff, tolerance for stopping criteria, whether print out logs or not
    classifier.fit(train_X, train_y)
    test_X, test_y = load_data(data_path = '20news-bydate/20news-test-tfidf.txt')
    predicted_y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_y = predicted_y, expected_y = test_y)
    print('Accuracy: ', accuracy)

if __name__ = 'main':
    classifying_with_linear_SVMs()    