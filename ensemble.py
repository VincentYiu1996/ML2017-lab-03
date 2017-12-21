import pickle
import numpy as np

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier=weak_classifier
        self.n_weakers_limit=n_weakers_limit

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        self.w=np.ones(X.shape[0])/X.shape[0]
        # self.clf.fit(X,y,sample_weight=self.w)
        self.clf=[]
        self.num=10
        self.error=[]
        self.alpha=[]

        #基本类器
        for i in range(self.num):
            self.clf.append(self.weak_classifier(max_depth=self.n_weakers_limit))
            # print(self.w)
            self.clf[i].fit(X, y, sample_weight=self.w)

            #calc error
            e=0
            G=self.clf[i].predict(X)
            for j in range(X.shape[0]):
                if G[j] != y[j]:
                    e += self.w[j]
            self.error.append(e/X.shape[0])

            self.alpha.append(0.5*np.log((1-self.error[i])/self.error[i]))
            Z=0
            for j in range(X.shape[0]):
                Z+=self.w[j]*np.exp(-self.alpha[i]*y[j]*G[j])

            self.w=self.w/Z*np.exp(-self.alpha[i]*y[j]*G[j])



    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        score=0
        for i in range(self.num):
            score+=self.alpha[i]*self.clf[i].predict(X)
        return score

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        G=0
        for i in range(self.num):
            G+=self.alpha[i]*self.clf[i].predict(X)



        return np.sign(G)


    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
