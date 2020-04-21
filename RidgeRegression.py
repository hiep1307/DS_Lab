import numpy as np


def normalize(X):
    X_max = np.array([[np.max(X[line, :]) for i in range(X.shape[1]) ] for line in range(X.shape[0])])
    X_min = np.array([[np.min(X[line, :]) for _ in range(X.shape[1])] for line in range(X.shape[0])])
    X_normalize = (X - X_min) / (X_max - X_min)
    return X_normalize


class RidgeRegression:
    def __init__(self):
        return
    def fit(self, X_train, Y_train, LAMBDA):
        assert len(X_train.shape) == 2 and X_train.shape[0] == Y_train.shape[0]
        return np.linalg.pinv(X_train.transpose().dot(X_train) + np.identity(X.shape[1]) * LAMBDA).dot(X_train.T).dot(Y_train)
    def fit_gradient_descent(self, X_train, Y_train, LAMBDA, learning_rate, max_num_epoch = 100, batch_size = 128):
        W = np.array([np.random.randn(X_train.shape[1])]).T
        last_loss = 10e+8
        arr = np.array(range(X_train.shape[0]))
        for epoch in range(max_num_epoch):
            np.random.shuffle(arr)
            X_train = X_train[arr]
            Y_train = Y_train[arr]
            total_minibatch = int(np.ceil(X_train.shape[0]/batch_size))
            for i in range(total_minibatch):
                index = i * batch_size
                X_train_sub = X_train[index : index + batch_size]
                Y_train_sub = Y_train[index : index + batch_size]
                gradient = X_train_sub.transpose().dot(X_train_sub.dot(W) - Y_train_sub) + LAMBDA * W
                W -= learning_rate * gradient
            new_loss = self.compute_RSS(Y_train, self.predict(W, X_train))
            if (np.abs(new_loss - last_loss) <= 10e-5):
                break
            last_loss = new_loss
        return W
    def predict(self, W, X_train):
        return X_train.dot(W)
    def compute_RSS(self, Y, Y_predict):
        return 1. / Y.shape[0] * np.sum((Y - Y_predict) ** 2)
    def get_the_best_LAMBDA(self, X_train, Y_train):
        def cross_validation(num_folds, LAMBDA):
            row_ids = np.array(range(X_train.shape[0]))
            valid_ids = np.split(row_ids[: len(row_ids) - len(row_ids) % num_folds], num_folds)
            valid_ids[-1] = np.append(valid_ids[-1], row_ids[len(row_ids) - len(row_ids) % num_folds : ])
            train_ids = [[k for k in row_ids if k not in valid_ids[i]] for i in range(num_folds)]
            sum_RSS = 0
            for i in range(num_folds):
                valid_part = {'X': X_train[valid_ids[i]], 'Y': Y_train[valid_ids[i]]}
                train_part = {'X': X_train[train_ids[i]], 'Y': Y_train[train_ids[i]]}
                W = self.fit(train_part['X'], train_part['Y'], LAMBDA)
                sum_RSS += self.compute_RSS(valid_part['Y'], self.predict(W, valid_part['X']))
            return sum_RSS / num_folds
        def range_scan(best_LAMBDA, minimum_RSS, LAMBDA_values):
            for LAMBDA in LAMBDA_values:
                average_RSS = cross_validation(num_folds = 5, LAMBDA= LAMBDA)
                if(average_RSS < minimum_RSS):
                    minimum_RSS = average_RSS
                    best_LAMBDA = LAMBDA
            return best_LAMBDA, minimum_RSS

        best_LAMBDA, minimum_RSS = range_scan(best_LAMBDA = 0, minimum_RSS = 1000 ** 2, LAMBDA_values = range(50))
        LAMBDA_values = [k * 1. / 1000 for k in range((best_LAMBDA - 1) * 1000, (best_LAMBDA + 1) * 1000)]
        best_LAMBDA, minimum_RSS = range_scan(best_LAMBDA = best_LAMBDA, minimum_RSS = minimum_RSS, LAMBDA_values = LAMBDA_values)
        return best_LAMBDA



X = []
Y = []
path= "./data.txt"
with open(path,"r") as f:
    for line in f.readlines():
        p= [ float(item) for item in line.split()]
        X.append(p[1:-1])
        Y.append(p[-1])
X= np.array(X, dtype= np.float64)
X= normalize(X)
one = np.ones((np.shape(X)[0],1))
X= np.concatenate((one, X), axis = 1)
Y= np.array([Y]).T 
X_train = X[:50]
Y_train = Y[:50]
X_test = X[50:]
Y_test = Y[50:]
# W = np.random.randn(X.shape[1])

# print(X.shape, Y.shape, W.shape)
test = RidgeRegression()
LAMBDA = test.get_the_best_LAMBDA(X_train, Y_train)
W = test.fit_gradient_descent(X_train, Y_train, LAMBDA, learning_rate= 0.01, max_num_epoch= 100, batch_size= 10)
Y_predict = test.predict(W, X_test)
print("RSS= ", test.compute_RSS(Y_test, Y_predict))