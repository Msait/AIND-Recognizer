import math
import statistics
import warnings

import numpy as np
import operator
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def print(self, s):
        if self.verbose:
            print(s)

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection based on BIC scores

        BICs = {}
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n_components)
                N = len(self.X) # number of datapoints (features)
                p = n_components * (n_components - 1) + 2 * N * n_components # number of parameters: n*(n-1) + 2*d*n
                logL = model.score(self.X, self.lengths)
                BIC = -2 * logL + p * np.log10(N)
                BICs[BIC] = model
            except:
                None
        # minimize BIC
        return min(BICs.items(), key=operator.itemgetter(0))[1]

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion
                                            
    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        DICs = {}
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            model = self.base_model(n_components)
            try:
                logL = model.score(self.X, self.lengths)
            except:
                continue

            anti_evidences = []

            # sum log(P(X(all except current word)))
            for word, (X, lengths) in self.hwords.items():
                if word == self.this_word:
                    continue
                # evidence for competing class
                try:
                    anti_evidences.append(model.score(X, lengths))
                except:
                    None

            DIC = logL - np.average(anti_evidences)
            DICs[DIC] = model

        # maximize DIC
        return max(DICs.items(), key= operator.itemgetter(0))[1]

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        # raise NotImplementedError
        # max of loop [min_n_components .. man_n_components] -> logL
        
        averages = {}
        # try to find best n_components 
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            k = min(3, len(self.sequences))
            split_method = KFold(k)
            score_set = []
        
            # split data on folds
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                try:
                    self.print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))
                    test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                    train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                    self.X = train_X
                    self.lengths = train_lengths
                    model = self.base_model(n_components)
                    # compute score`
                    score_set.append(model.score(test_X, test_lengths))
                except:
                    None
            
            # find average
            avg = np.average(score_set)
            averages[avg] = n_components
            self.print("Scores for {} n_components is {}. K-Folds {}. Average {} for {} n_components\n".format(n_components, score_set, k, avg, n_components))

        # find max (score > average)
        best_n_component_by_score = max(averages.items(), key=operator.itemgetter(0))[1]
        self.X, self.lengths = self.hwords[self.this_word]
        return self.base_model(best_n_component_by_score)
