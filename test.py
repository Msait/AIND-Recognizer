from  my_model_selectors import SelectorCV
from my_model_selectors import SelectorBIC
from my_model_selectors import SelectorDIC
from my_model_selectors import SelectorConstant
from my_recognizer import recognize
from asl_utils import show_errors
import numpy as np

from asl_data import AslDb

asl = AslDb() # initializes the database
words_to_train = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']
import timeit

# TODO: Implement SelectorCV in my_model_selector.py
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']


training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
sequences = training.get_all_sequences()
Xlengths = training.get_all_Xlengths()

def build_features_norm():
    df_means = asl.df.groupby('speaker').mean()
    df_std = asl.df.groupby('speaker').std()

    asl.df['left-x-std'] = asl.df['speaker'].map(df_std['left-x'])
    asl.df['left-y-std'] = asl.df['speaker'].map(df_std['left-y'])
    asl.df['right-x-std'] = asl.df['speaker'].map(df_std['right-x'])
    asl.df['right-y-std'] = asl.df['speaker'].map(df_std['right-y'])

    asl.df['left-x-mean'] = asl.df['speaker'].map(df_means['left-x'])
    asl.df['left-y-mean'] = asl.df['speaker'].map(df_means['left-y'])
    asl.df['right-x-mean'] = asl.df['speaker'].map(df_means['right-x'])
    asl.df['right-y-mean'] = asl.df['speaker'].map(df_means['right-y'])

    asl.df['norm-lx'] = (asl.df['left-x'] - asl.df['left-x-mean']) / asl.df['left-x-std']
    asl.df['norm-ly'] = (asl.df['left-y'] - asl.df['left-y-mean']) / asl.df['left-y-std']
    asl.df['norm-rx'] = (asl.df['right-x'] - asl.df['right-x-mean']) / asl.df['right-x-std']
    asl.df['norm-ry'] = (asl.df['right-y'] - asl.df['right-y-mean']) / asl.df['right-y-std']
    return ['norm-rx', 'norm-ry', 'norm-lx', 'norm-ly']

def build_features_polar():
    a1 = (asl.df['right-x'] - asl.df['nose-x'])
    b1 = (asl.df['right-y'] - asl.df['nose-y'])
    asl.df['polar-rr'] = np.hypot(a1, b1)

    a2 = (asl.df['left-x'] - asl.df['nose-x'])
    b2 = (asl.df['left-y'] - asl.df['nose-y'])
    asl.df['polar-lr'] = np.hypot(a2, b2)

    asl.df['polar-rtheta'] = np.arctan2(a1, b1)
    asl.df['polar-ltheta'] = np.arctan2(a2, b2)

    return ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']

def build_features_delta():
    asl.df['delta-rx'] = asl.df['right-x'].diff().fillna(0)
    asl.df['delta-ry'] = asl.df['right-y'].diff().fillna(0)
    asl.df['delta-lx'] = asl.df['left-x'].diff().fillna(0)
    asl.df['delta-ly'] = asl.df['left-y'].diff().fillna(0)
    return ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']

def build_custom_feature_1_dist():
    asl.df['dist-x'] = asl.df['right-x'] - asl.df['left-x']
    asl.df['dist-y'] = asl.df['right-y'] - asl.df['left-y']
    asl.df['dist-grnd-x'] = asl.df['grnd-rx'] - asl.df['grnd-lx']
    asl.df['dist-grnd-y'] = asl.df['grnd-ry'] - asl.df['grnd-ly']
    return ['dist-x', 'dist-y', 'dist-grnd-x', 'dist-grnd-y']

def build_custom_feature_2_polar_normalized():
    df_means = asl.df.groupby('speaker').mean()
    df_std = asl.df.groupby('speaker').std()
    # polar normalization arm position
    polar_rtheta_mean = asl.df['speaker'].map(df_means['polar-rtheta'])
    polar_rtheta_std = asl.df['speaker'].map(df_std['polar-rtheta'])
    asl.df['norm-polar-rtheta'] = (asl.df['polar-rtheta'] - polar_rtheta_mean) / polar_rtheta_std

    polar_ltheta_mean = asl.df['speaker'].map(df_means['polar-ltheta'])
    polar_ltheta_std = asl.df['speaker'].map(df_std['polar-ltheta'])
    asl.df['norm-polar-ltheta'] = (asl.df['polar-rtheta'] - polar_ltheta_mean) / polar_ltheta_std

    # polar normalization arm length
    polar_rr_mean = asl.df['speaker'].map(df_means['polar-rr'])
    polar_rr_std = asl.df['speaker'].map(df_std['polar-rr'])
    asl.df['norm-polar-rr'] = (asl.df['polar-rr'] - polar_rr_mean) / polar_rr_std

    polar_lr_mean = asl.df['speaker'].map(df_means['polar-lr'])
    polar_lr_std = asl.df['speaker'].map(df_std['polar-lr'])
    asl.df['norm-polar-lr'] = (asl.df['polar-lr'] - polar_lr_mean) / polar_lr_std
    return ['norm-polar-rtheta', 'norm-polar-ltheta', 'norm-polar-rr', 'norm-polar-lr']

def test_selectorCV():
    for word in words_to_train:
        start = timeit.default_timer()
        model = SelectorCV(sequences, Xlengths, word, verbose=False,
                           min_n_components=2, max_n_components=15, random_state=14).select()
        end = timeit.default_timer() - start
        if model is not None:
            print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
        else:
            print("Training failed for {}".format(word))


def test_selectorBIC():
    for word in words_to_train:
        start = timeit.default_timer()
        model = SelectorBIC(sequences, Xlengths, word,
                            min_n_components=2, max_n_components=15, random_state=14).select()
        end = timeit.default_timer() - start
        if model is not None:
            print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
        else:
            print("Training failed for {}".format(word))


def test_selectorDIC():
    for word in words_to_train:
        start = timeit.default_timer()
        model = SelectorDIC(sequences, Xlengths, word,
                            min_n_components=2, max_n_components=15, random_state=14).select()
        end = timeit.default_timer() - start
        if model is not None:
            print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
        else:
            print("Training failed for {}".format(word))

def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word,
                        n_constant=3).select()
        model_dict[word]=model
    return model_dict

def show_errors_for(guesses, selector_names, test_set):
    for (guess, selector_name) in zip(guesses, selector_names):
        print("Selector: {}".format(selector_name))
        show_errors(guess, test_set)


import threading
class RecognizeThread(threading.Thread):
    def __init__(self, name=None, features=None, selectors=None):
        threading.Thread.__init__(self)
        self.name = name
        self.features = features
        self.selectors = selectors
        self.test_set = asl.build_test(features)
        self.start_train = timeit.default_timer()
        self.models = [self.train_all_words(features, s) for s in selectors]
        self.end_train = timeit.default_timer() - self.start_train
        self.selector_name_list = ["SelectorCV", "SelectorBIC", "SelectorDIC"]

    def show_errors_for(self, guesses, selector_names, test_set):
        for (guess, selector_name) in zip(guesses, selector_names):
            print("Selector: {}".format(selector_name))
            show_errors(guess, test_set)

    def run(self):
        print("Start recognition...")
        probabilities_guesses_list = [recognize(m, self.test_set) for m in self.models]
        print("Show errors for {} with {}".format(self.selector_name_list, self.features))
        self.show_errors_for([p_g[1] for p_g in probabilities_guesses_list], self.selector_name_list, self.test_set)
        print("Train time: {}".format(self.end_train - self.start_train))


    def train_all_words(self, features, model_selector):
        training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
        sequences = training.get_all_sequences()
        Xlengths = training.get_all_Xlengths()
        model_dict = {}
        for word in training.words:
            model = model_selector(sequences, Xlengths, word,
                                   n_constant=3).select()
            model_dict[word] = model
        return model_dict


def test_recognize():

    # TODO Choose a feature set and model selector
    features = features_ground  # change as needed

    features_norm = build_features_norm()
    features_polar = build_features_polar()
    features_delta = build_features_delta()
    features_custom_1 = build_custom_feature_1_dist()
    features_custom_2 = build_custom_feature_2_polar_normalized()
    model_selector = SelectorConstant  # change as needed

    # TODO Recognize the test set and display the result with the show_errors method

    # thread_feature_norm = RecognizeThread("feature norm", features_norm, [SelectorCV, SelectorBIC, SelectorDIC])
    # thread_feature_polar = RecognizeThread("feature polar", features_polar, [SelectorCV, SelectorBIC, SelectorDIC])
    # thread_feature_delta = RecognizeThread("feature delta", features_delta, [SelectorCV, SelectorBIC, SelectorDIC])
    # thread_feature_custom_1 = RecognizeThread("feature custom_1", features_custom_1, [SelectorCV, SelectorBIC, SelectorDIC])
    # thread_feature_custom_2 = RecognizeThread("feature custom_2", features_custom_2, [SelectorCV, SelectorBIC, SelectorDIC])
    #
    # thread_feature_norm.start()
    # thread_feature_polar.start()
    # thread_feature_delta.start()
    # thread_feature_custom_1.start()
    # thread_feature_custom_2.start()
    #
    # thread_feature_norm.join()
    # thread_feature_polar.join()
    # thread_feature_delta.join()
    # thread_feature_custom_1.join()
    # thread_feature_custom_2.join()

    thread_feature_total = RecognizeThread("feature norm + feature polar + feature custom_2",
                                              features_norm + features_polar + features_delta + features_custom_2,
                                              [SelectorCV, SelectorBIC, SelectorDIC])
    thread_feature_total.start()
    thread_feature_total.join()

def test_selectors():
    print("DIC")
    test_selectorDIC()
    print("BIC")
    test_selectorBIC()
    print("CV")
    test_selectorCV()



# test_selectors()

test_recognize()