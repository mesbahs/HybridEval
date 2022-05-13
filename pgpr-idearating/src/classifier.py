import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import sklearn.decomposition
from sklearn.preprocessing import FunctionTransformer
import re
import nltk
import gensim
from nltk.corpus import stopwords
from nltk import word_tokenize
from IPython.core.interactiveshell import InteractiveShell
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
import torch
import transformers as ppb
import argparse



import numpy as np
import torch
from pytorch_transformers import BertModel, BertTokenizer


from IPython import embed
from sklearn.model_selection import RandomizedSearchCV
class classifier_method:
    def __init__(self):
        print("model initialized")

    def classification_report_csv(self,report,evaluation_file):
        report_data = []
        lines = report.split('\n')
        lines = [t for t in lines if len(t) > 1]
        for line in lines[2:-3]:
            row = {}
            row_data = line.split('      ')
            row_data = [t for t in row_data if len(t) > 1]
            row['class'] = row_data[0]
            row['precision'] = float(row_data[1])
            row['recall'] = float(row_data[2])
            row['f1_score'] = float(row_data[3])
            row['support'] = float(row_data[4])
            report_data.append(row)
        dataframe = pd.DataFrame.from_dict(report_data)
        dataframe.to_csv(evaluation_file, index = False,mode='a')

    # def parse_args():
    #     parser = argparse.ArgumentParser(
    #         description="naive bayes")
    #     parser.add_argument("--inputfile",
    #                         type=str,
    #                         required=True,
    #                         help="inputfile reviews")
    #     parser.add_argument("--sup_rate",
    #                         default=8,
    #                         type=int,
    #                         help="supervision rate")
    #     parser.add_argument("--evaluation_file",
    #                         type=str,
    #                         required=True,
    #                         help="evaluation result after each iteration")
    #
    #     args = parser.parse_args()
    #     return args

    def clean_text(self, text):
        text=text.replace('<p>','')
        text = text.replace('</p>', '')
        text = text.replace('</h4>', '')
        text = text.replace('<h4>', '')
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        # lower text
        text = text.lower()
        # stemmer = SnowballStemmer("english")
        # text = BAD_SYMBOLS_RE.sub('',text)
        text = REPLACE_BY_SPACE_RE.sub(' ', text)
        # tokenize text
        text = text.split(" ")
        # remove stop words
        stop = stopwords.words('english')
        text = [x for x in text if x not in stop]
        # remove words with only one letter or empty
        text = [t for t in text if len(t) > 1]
        # stems = []
        # for t in text:
        #     stems.append(stemmer.stem(t))
        # join all
        text = " ".join(text)
        return (text)

    def w2v_tokenize_text(self,text):
        tokens = []
        for sent in nltk.sent_tokenize(text, language='english'):
            for word in nltk.word_tokenize(sent, language='english'):
                if len(word) < 2:
                    continue
                tokens.append(word)
        return tokens

    def word_averaging(self,wv, words):
        all_words, mean = set(), []

        for word in words:
            if isinstance(word, np.ndarray):
                mean.append(word)
            elif word in wv.vocab:
                mean.append(wv.syn0norm[wv.vocab[word].index])
                all_words.add(wv.vocab[word].index)

        if not mean:
            logging.warning("cannot compute similarity with no input %s", words)
            # FIXME: remove these examples in pre-processing
            return np.zeros(wv.vector_size, )

        mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
        return mean

    def word_averaging_list(self,wv, text_list):
        return np.vstack([self.word_averaging(wv, post) for post in text_list])

    def svm_review_main(self):

        svm_pipeline = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-5, random_state=30, max_iter=5, tol=None)),
                       ])
        return svm_pipeline

   
    def logreg_review_main(self):

        logreg = Pipeline([('vect', CountVectorizer()),
                           ('tfidf', TfidfTransformer()),
                           ('clf', LogisticRegression(n_jobs=1, C=1e7, multi_class='auto', solver='newton-cg')),
                           ])
        return logreg



    def logreg_embedding(self,X_train, X_test):
        word2model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin',
                                                                     binary=True)
        word2model.init_sims(replace=True)
        test_tokenized = X_test.apply(lambda r: self.w2v_tokenize_text(r)).values
        train_tokenized = X_train.apply(lambda r: self.w2v_tokenize_text(r)).values

        X_train_word_average = self.word_averaging_list(word2model, train_tokenized)
        X_test_word_average = self.word_averaging_list(word2model, test_tokenized)





        logreg = LogisticRegression(n_jobs=1, C=1e5)

        return logreg,X_train_word_average,X_test_word_average


    def reg_embedding(self,X_train, X_test):
        word2model = gensim.models.KeyedVectors.load_word2vec_format('/Users/smesbah/Downloads/GoogleNews-vectors-negative300.bin',
                                                                     binary=True)
        word2model.init_sims(replace=True)
        test_tokenized = X_test.apply(lambda r: self.w2v_tokenize_text(r)).values
        train_tokenized = X_train.apply(lambda r: self.w2v_tokenize_text(r)).values

        X_train_word_average = self.word_averaging_list(word2model, train_tokenized)
        X_test_word_average = self.word_averaging_list(word2model, test_tokenized)





        # logreg = AdaBoostRegressor(random_state=0, n_estimators=200)
        param_dist = {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
            'loss': ['linear', 'square', 'exponential']
        }

        logreg = RandomizedSearchCV(AdaBoostRegressor(),
                                    param_distributions=param_dist,
                                    cv=3,
                                    n_iter=10,
                                    n_jobs=-1)

        return logreg,X_train_word_average,X_test_word_average


    def reg_embedding_2(self,X_train, X_test):




        model_class, tokenizer_class, pretrained_weights = (
            ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

        # Load pretrained model/tokenizer
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        tokenized = X_train.apply(
            (lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=500, truncation=True)))

        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
        attention_mask = np.where(padded != 0, 1, 0)
        attention_mask.shape
        input_ids = torch.tensor(padded)
        attention_mask = torch.tensor(attention_mask)

        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)
        X_train = last_hidden_states[0][:, 0, :].numpy()

        tokenized = X_test.apply(
            (lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=500, truncation=True)))

        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
        attention_mask = np.where(padded != 0, 1, 0)
        attention_mask.shape
        input_ids = torch.tensor(padded)
        attention_mask = torch.tensor(attention_mask)

        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)
        X_test = last_hidden_states[0][:, 0, :].numpy()





        # logreg = AdaBoostRegressor(random_state=0, n_estimators=200)
        param_dist = {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
            'loss': ['linear', 'square', 'exponential']
        }

        logreg = RandomizedSearchCV(AdaBoostRegressor(),
                                    param_distributions=param_dist,
                                    cv=3,
                                    n_iter=10,
                                    n_jobs=-1)

        return logreg,X_train,X_test


    def reg_embedding_binary(self,X_train, X_test):




        model_class, tokenizer_class, pretrained_weights = (
            ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

        # Load pretrained model/tokenizer
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        tokenized = X_train.apply(
            (lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=500, truncation=True)))

        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
        attention_mask = np.where(padded != 0, 1, 0)
        attention_mask.shape
        input_ids = torch.tensor(padded)
        attention_mask = torch.tensor(attention_mask)

        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)
        X_train = last_hidden_states[0][:, 0, :].numpy()

        tokenized = X_test.apply(
            (lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=500, truncation=True)))

        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
        attention_mask = np.where(padded != 0, 1, 0)
        attention_mask.shape
        input_ids = torch.tensor(padded)
        attention_mask = torch.tensor(attention_mask)

        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)
        X_test = last_hidden_states[0][:, 0, :].numpy()




        logreg=LogisticRegression(n_jobs=1, C=1e7, multi_class='auto', solver='newton-cg')

        return logreg,X_train,X_test
    def regressor(self,):


        logreg = Pipeline([('vect', CountVectorizer()),
                           ('tfidf', TfidfTransformer()),
                           ('clf', AdaBoostRegressor(random_state=0, n_estimators=200)),
                           ])

        


        return logreg


    