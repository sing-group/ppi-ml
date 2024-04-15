from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class PPIIBM(BaseEstimator, ClassifierMixin):
    """
    PPIIBM (Pair Prediction by Item Identification Baseline Model)

    This is a simple baseline model for binary classification of pairs of items: (item-1, item-2 -> Y/N)
    The features of each item is assumed to be concatenated in X, that is, each pair contains 2*features-per-item features.

    This model exploits positivity-related bias that may arise, for example:
        - The same item appears in multiple pairs (pairs are not repeated, but individual items are).
        - There are the same number of positive and negative pairs (the dataset seems balanced). But:
        - Each item is very unbalanced, that is, it has many positive or many negative relations. This 
          unbalance has to be in both ways: some items unbalanced towards positive and some items 
          unbalanced towards negative.
          
    In this scenario, when validating models via random splits *at pair level* (CV, Train-Test, etc),
    the model can simply recognize one of the items of the pair to be predicted and predict based on the apriori
    probability of such item to be in a positive or negative pair, found during train.

    This model, simply trains by memorizing the apriori probability of each
    individual item. The more the unbalance at item level, the more -false- performace this model will get when
    evaluated in random split settings. More concretely:

    singleItemMode = True (only works with the positivity of the first item)
        Prediction = 1  when (positive_pairs_of_item1_in_train) /
                            (total_pairs_of_item1_in_train) > 0.5
                    0, otherwise

                    ie, it predicts the prevalence in the train dataset focusing only in the first item
        
    bothItemsMode = False (pools the positivity of both items in the interaction)
        Prediction = 1  when ((positive_pairs_of_item1_in_train/total_pairs_of_item1_in_train) + 
                             (positive_pairs_of_item2_in_train / total_pairs_of_item2_in_train)) / 2 > 0.5
                    0, otherwise

                    ie, it predicts the prevalence in the train dataset focusing only in both items

    The bias can be avoided by splitting at item level, that is, the item to be predicted should be unseen,
    it should not be found during train.
    """

    def __init__(self, singleItemMode=True, print_debug_messages=False) -> None:
        super().__init__()
        self.singleItemMode = singleItemMode
        self.print_debug_messages = print_debug_messages

    def fit(self, X, y):
        
        """ 
        A dictionary with
            - item -> [positive_pairs_count, total_pairs_count]
        """
        self.items_memory = {}

        self.total_pairs_in_train = X.shape[0]
        self.positive_pairs_in_train = np.sum(y == 1)        
        
        for pair, y in zip(X, y):            
            self.__updateCountsForPairItems(pair, y)

        if self.print_debug_messages:
            print(f'PPIIBM (first item mode: {self.singleItemMode}). Fitted. Total pairs: {self.total_pairs_in_train} "; Positive: {self.positive_pairs_in_train}; Number of distinct items {len(self.items_memory)}')

        return self

    def predict(self, X):
        
        predictions = np.zeros(X.shape[0])
        predictions_with_items_found = 0
        
        for i, pair in enumerate(X):
            
            item1_key, item2_key = self.__getPairItemsKeys(pair)


            if self.singleItemMode == True:
                ##### naive mode: only take the first item, do not interactions
                if item1_key in self.items_memory:
                    predictions_with_items_found += 1
                    predictions[i] = 1 if self.__getItemTrainPositivity(item1_key) > 0.5 else 0

                else:
                    # Not found any item in train
                    predictions[i] = 1 if self.__getOverallTrainPositivity() > 0.5 else 0

            else:
                if item1_key in self.items_memory or item2_key in self.items_memory:
                    #### interaction mode: it takes into accound both items
            
                    # Found at least one item in train
                    predictions_with_items_found += 1
                    predictions[i] = 1 if self.__getItemsTrainPositivity(item1_key, item2_key) > 0.5 else 0
                else:
                    # Not found any item in train
                    predictions[i] = 1 if self.__getOverallTrainPositivity() > 0.5 else 0

        if self.print_debug_messages:
            print(f'PPIIBM. Predicted. Made {predictions.size} predictions. Identified one or both items in train in {predictions_with_items_found} predictions')

        return predictions

    def __getItemTrainPositivity(self, item_key):
        positive_pairs_of_item1_in_train, total_pairs_of_item1_in_train = self.__getItemCountsInTrain(item_key)
        return (positive_pairs_of_item1_in_train / total_pairs_of_item1_in_train)

    def __getItemsTrainPositivity(self, item1_key, item2_key):
        positive_pairs_of_item1_in_train, total_pairs_of_item1_in_train = self.__getItemCountsInTrain(item1_key)
        positive_pairs_of_item2_in_train, total_pairs_of_item2_in_train = self.__getItemCountsInTrain(item2_key)

        #return (positive_pairs_of_item1_in_train + positive_pairs_of_item2_in_train) / (total_pairs_of_item1_in_train + total_pairs_of_item2_in_train)
        if total_pairs_of_item1_in_train > 0 and total_pairs_of_item2_in_train > 0:
            return ((positive_pairs_of_item1_in_train/total_pairs_of_item1_in_train) + (positive_pairs_of_item2_in_train / total_pairs_of_item2_in_train)) / 2
        elif total_pairs_of_item2_in_train > 0:
            return (positive_pairs_of_item2_in_train / total_pairs_of_item2_in_train)
        else:
            return (positive_pairs_of_item1_in_train / total_pairs_of_item1_in_train)
    

    
    def __getOverallTrainPositivity(self):
        return self.positive_pairs_in_train / self.total_pairs_in_train

    def __getItemCountsInTrain(self, item_key):
        if item_key in self.items_memory:
            return self.items_memory[item_key][0], self.items_memory[item_key][1]
        else:
            return 0, 0

    def __updateCountsForPairItems(self, pair, pair_y):
        item1_key, item2_key = self.__getPairItemsKeys(pair)
        self.__updateCountsForItemInPair(item1_key, pair_y)
        self.__updateCountsForItemInPair(item2_key, pair_y)
    
    def __getPairItemsKeys(self, pair):
        return tuple(pair[0:int(pair.size/2)]), tuple(pair[int(pair.size/2):pair.size])

    def __updateCountsForItemInPair(self, item_key, pair_y):
        if item_key not in self.items_memory:
            self.items_memory[item_key] = [0, 0]
        self.items_memory[item_key] = [self.items_memory[item_key][0]+(1 if pair_y == 1 else 0), self.items_memory[item_key][1]+1]
