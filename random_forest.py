from collections import Counter
import threading
from decision_tree import DecisionTree
import numpy as np


class RandomForest:
    '''
    A class that implements Random Forest algorithm from scratch.
    '''
    def __init__(self, num_trees=25, min_samples_split=5, max_depth=4):
        self.num_trees = num_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        # Will store individually trained decision trees
        self.decision_trees = []
        self._lock = threading.Lock()
        
    @staticmethod
    def _sample(X, y):
        '''
        Helper function used for boostrap sampling.
        
        :param X: np.array, features
        :param y: np.array, target
        :return: tuple (sample of features, sample of target)
        '''
        n_rows, n_cols = X.shape
        X_bag, y_bag = X.copy(), y.copy()
        # Sample with replacement
        samples = np.random.choice(a=n_rows, size=n_rows, replace=True)
        X_bag = X_bag[samples]
        y_bag = y_bag[samples]
        # features = np.random.choice(a=n_cols, size=n_cols - int(np.sqrt(n_cols)), replace=False)
        # X_bag[:, features] = 0

        return X_bag, y_bag
    
    def _fit_tree(self, clf, X, y):
        clf.fit(X, y)
        with self._lock:
            self.decision_trees.append(clf)

    # def fit(self, X, y):
    #     # Reset decision_trees
    #     self.decision_trees = []
    #     threads = []
    #     for _ in range(self.num_trees):
    #         clf = DecisionTree(
    #             min_samples_split=self.min_samples_split,
    #             max_depth=self.max_depth
    #         )
    #         _X, _y = self._sample(X, y)
    #         t = threading.Thread(target=self._fit_tree, args=(clf, _X, _y))
    #         t.start()
    #         threads.append(t)

    #     # Wait for all threads to finish
    #     for t in threads:
    #         t.join()


    def fit(self, X, y):
        # Reset
        self.decision_trees = []
            
        # Build each tree of the forest
        num_built = 0
        while num_built < self.num_trees:
            
            clf = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth
            )
            # Obtain data sample
            _X, _y = self._sample(X, y)
            # Train
            clf.fit(_X, _y)
            # Save the classifier
            self.decision_trees.append(clf)
            num_built += 1
            

        
    def predict(self, X):
        '''
        Predicts class labels for new data instances.
        
        :param X: np.array, new instances to predict
        :return: 
        '''
        # Make predictions with every tree in the forest
        y = []
        for tree in self.decision_trees:
            y.append(tree.predict(X))
            # print('tree')
        
        # Reshape so we can find the most common value
        y = np.swapaxes(a=y, axis1=0, axis2=1)
        
        # Use majority voting for the final prediction
        predictions = []
        for preds in y:
            counter = Counter(preds)
            predictions.append(counter.most_common(1)[0][0])
        return predictions
