import numpy as np
from node import Node
from collections import Counter

class DecisionTree:
    '''
    Class which implements a decision tree classifier algorithm.
    '''
    def __init__(self, min_samples_split=5, max_depth=4):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        
    @staticmethod
    def _entropy(s):
        '''
        Helper function, calculates entropy from an array of integer values.
        
        :param s: list
        :return: float, entropy value
        '''
        # Convert to integers to avoid runtime errors
        counts = np.bincount(np.array(s, dtype=np.int64))
        # Probabilities of each class label
        percentages = counts / len(s)

        # Caclulate entropy
        entropy = 0
        for pct in percentages:
            if pct > 0:
                entropy += pct * np.log2(pct)
        return -entropy
    
    def _information_gain(self, parent, left_child, right_child):
        '''
        Helper function, calculates information gain from a parent and two child nodes.
        
        :param parent: list, the parent node
        :param left_child: list, left child of a parent
        :param right_child: list, right child of a parent
        :return: float, information gain
        '''
        num_left = len(left_child) / len(parent)
        num_right = len(right_child) / len(parent)
        
        # One-liner which implements the previously discussed formula
        return self._entropy(parent) - (num_left * self._entropy(left_child) + num_right * self._entropy(right_child))
    
    def _best_split(self, X, y):
        '''
        Helper function, calculates the best split for given features and target
        
        :param X: np.array, features
        :param y: np.array or list, target
        :return: dict
        '''
        best_split = {}
        best_info_gain = -1
        n_rows, n_cols = X.shape
        
        # For every dataset feature
        for f_idx in range(n_cols):
            # print(X[0,:f_idx])
            X_curr = X[:, f_idx]

            # For every unique value of that feature
            for threshold in np.unique(X_curr):
                # Construct a dataset and split it to the left and right parts
                # Left part includes records lower or equal to the threshold
                # Right part includes records higher than the threshold
                df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
                if not isinstance(threshold, str):
                    df_left = np.array([row for row in df if row[f_idx] <= threshold])
                    df_right = np.array([row for row in df if row[f_idx] > threshold])
                else:
                    df_left = np.array([row for row in df if row[f_idx] == threshold])
                    df_right = np.array([row for row in df if row[f_idx] != threshold])

                # Do the calculation only if there's data in both subsets
                if len(df_left) > 0 and len(df_right) > 0:
                    # Obtain the value of the target variable for subsets
                    y = df[:, -1]
                    y_left = df_left[:, -1]
                    y_right = df_right[:, -1]

                    # Caclulate the information gain and save the split parameters
                    # if the current split is better than the previous best
                    gain = self._information_gain(y, y_left, y_right)
                    if gain > best_info_gain:
                        # print(gain)
                        best_split = {
                            'feature_index': f_idx,
                            'threshold': threshold,
                            'df_left': df_left,
                            'df_right': df_right,
                            'gain': gain
                        }
                        best_info_gain = gain
        return best_split
    
    def _build(self, X, y, depth=0):
        '''
        Helper recursive function, used to build a decision tree from the input data.
        
        :param X: np.array, features
        :param y: np.array or list, target
        :param depth: current depth of a tree, used as a stopping criteria
        :return: Node
        '''
        n_rows, n_cols = X.shape
        
        # Check to see if a node should be leaf node
        if n_rows >= self.min_samples_split and depth <= self.max_depth:
            # Get the best split
            best = self._best_split(X, y)
            # print(best)
            # If the split isn't pure
            if len(best) > 0:
                if best['gain'] > 0:
                    # print('best gain')
                    # Build a tree on the left
                    left = self._build(
                        X=best['df_left'][:, :-1], 
                        y=best['df_left'][:, -1], 
                        depth=depth + 1
                    )
                    right = self._build(
                        X=best['df_right'][:, :-1], 
                        y=best['df_right'][:, -1], 
                        depth=depth + 1
                    )
                    return Node(
                        feature=best['feature_index'], 
                        threshold=best['threshold'], 
                        data_left=left, 
                        data_right=right, 
                        gain=best['gain']
                    )
        # Leaf node - value is the most common target value 
        return Node(
            value=Counter(y).most_common(1)[0][0]
        )
    
    def fit(self, X, y):
        '''
        Function used to train a decision tree classifier model.
        
        :param X: np.array, features
        :param y: np.array or list, target
        :return: None
        '''
        # Call a recursive function to build the tree
        self.root = self._build(X, y)
        
    def _predict(self, x, tree):
        '''
        Helper recursive function, used to predict a single instance (tree traversal).
        
        :param x: single observation
        :param tree: built tree
        :return: float, predicted class
        '''

        # Leaf node
        if tree.value != None:
            return tree.value
        feature_value = x[tree.feature]
        # print(feature_value, tree.threshold)
        if isinstance(feature_value, str):
            # Go to the left
            if feature_value == tree.threshold:
                return self._predict(x=x, tree=tree.data_left)
            
            # Go to the right
            if feature_value != tree.threshold:
                return self._predict(x=x, tree=tree.data_right)
        else:
            # Go to the left
            if feature_value <= tree.threshold:
                return self._predict(x=x, tree=tree.data_left)
            
            # Go to the right
            if feature_value > tree.threshold:
                return self._predict(x=x, tree=tree.data_right)

        
    def predict(self, X):
        '''
        Function used to classify new instances.
        
        :param X: np.array, features
        :return: np.array, predicted classes
        '''
        # Call the _predict() function for every observation
        # print('X')
        return [self._predict(x, self.root) for x in X]