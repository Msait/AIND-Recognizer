import warnings
from asl_data import SinglesData
import pandas as pd
import arpa

def get_adjecent_words(guesses, n_gram):
    return ' '.join(guesses[len(guesses)-n_gram:])

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    n_gram = 3
    lsm = arpa.loadf("slm/devel-lm-M{}.sri.lm".format(n_gram))
    lm = lsm[0]
    # TODO implement the recognizer

    # go foreach sentences:
    #   for each word in sentence:
    #      probability = {}
    #      for each train-word(model):
    #           score guess-word with model
    #           find in slm: logP of word that has predecessor of current guess-word.
    #           total_score = K * logP + logL
    #           probability[train-word] = total_score
    #
    #      # find max total_score
    #      import operator
    #      guess_word = max(probability.items(), key=operator.itemgetter(1))[0]
    #      guesses.append(guess_word)
    #      probabilities.append(probability)

    K = 50
    for test_X, test_Xlength in test_set.get_all_Xlengths().values():

        probability = {}
        for word, model in models.items():
            # calculate the scores for each model(word) and update the 'probabilities' list.
            try:
                logL = model.score(test_X, test_Xlength)
                if not guesses:
                    logP = lm.log_p('<s>')
                    score_start = K * logP + logL
                    logP = lm.log_p('</s>')
                    score_end = K * logP + logL
                    if score_end > score_start:
                        logP = lm.log_p('</s>')
                    else:
                        logP = lm.log_p('<s>')
                else:
                    w = get_adjecent_words(guesses, n_gram)
                    logP = lm.log_p(w)

                probability[word] = K * logP + logL
            except:
                probability[word] = float("-inf")
                pass

        import operator
        guess_word = max(probability.items(), key=operator.itemgetter(1))[0]

        guesses.append(guess_word)
        probabilities.append(probability)

    return probabilities, guesses
