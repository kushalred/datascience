import nltk
import pandas as pd
import numpy as np
from collections import defaultdict 
import re
import string

from custom_logic import CustomLogic
from config import Config


class Sentiment:
    """ 
        Sentiment class build and add new column of sentiment
    """

    def __init__(self, data_file):
        self.data = data_file
        self.logic = CustomLogic()

    def read_training_files(self):
        '''
        to read the training datasets from the input files for positive, 
        negative sentiments 
        '''
        positive_negative_scores = {}
        positive_scores = {}
        negative_scores = {}
        lemmatizer = nltk.stem.WordNetLemmatizer()
        with open(Config.TRANINGSET) as f:
            fileread = f.readlines()
            positive_negative_scores = {' '.join(x.split()[:-1]): float(x.split()[-1])
                                        for x in fileread}
        with open(Config.POSITIVEWORDS) as f1:
            positive_scores = f1.read()
        with open(Config.NEGITIVEWORDS) as f2:
            negative_scores = f2.read()
        with open(Config.INCWORDS) as f3:
            inc = f3.read()
        with open(Config.INVWORDS) as f4:
            inv = f4.read()
            # Mix 2 files
        with open(Config.AFINNWORDS) as f:
            fileread = f.readlines()
            pos_neg_score = {x.strip().split('\t')[0]: float(
                x.strip().split('\t')[1]) / 5.0 for x in fileread}
        for k, v in list(positive_negative_scores.items()):
            if k not in pos_neg_score:
                pos_neg_score[k] = float(v)
            else:
                pos_neg_score[k] = (pos_neg_score[k] + float(v)) / 2
        # Save the training set with lemmatized forms
        lemmatized_pos_neg = defaultdict(list)
        for k, v in list(pos_neg_score.items()):
            #k = k.decode('utf-8')
            tag = nltk.tag.pos_tag(nltk.word_tokenize(k))
            for x in tag:
                try:
                    lemmatized_pos_neg[lemmatizer.lemmatize(
                        x[0], get_wordnet_pos(x[1]))].append(v)
                except:
                    lemmatized_pos_neg[lemmatizer.lemmatize(k)].append(v)
                    
            training_scores = dict([(k, round(np.mean(v), 4))
                                    for k, v in list(lemmatized_pos_neg.items())])
        return training_scores, positive_scores, negative_scores, inc, inv

    def get_adj_score(self, adjective_variant, pos_neg_score, positive_scores, negative_scores):
        '''
        Function that takes an adjective variant which is a tuple of adjective word and its POS
        and returns the adjective score of the lemmatized form of the adjcetive from the 
        training files 
        '''
        lemmatizer = nltk.stem.WordNetLemmatizer()
        try:
            adjective = lemmatizer.lemmatize(adjective_variant[0].lower(), pre.get_wordnet_pos(
                adjective_variant[1]))  # Since we are searching only for adjectives
        except:
            adjective = lemmatizer.lemmatize(adjective_variant[0].lower())
        score = 0
        try:
            score = float(pos_neg_score[adjective])
        # use regex to avoid faulty matches for adjectives
        except:
            if adjective not in string.punctuation:
                adjective = adjective.replace('*', '').replace('+', '')
                pattern = r'\n' + adjective + '[ ]?\n'
                if len(re.findall(pattern, positive_scores)) >= 1:
                    score = 0.5
                elif len(re.findall(pattern, negative_scores)) >= 1:
                    score = -0.5

        else:
            pass
        return score

    def get_phrases_scores(self, df):        
        '''
        Function that takes a df as an input and returns a dictionary that has index as keys 
        and lists of phrases and their sentiment scores as values 
        '''
        unique_comments = df[Config.SurveyColumnName]
        pos_neg_score, positive_scores, negative_scores, inc, inv = self.read_training_files()
        modular_words = ['hope', 'need', 'would']
        negated_l = [x.strip() for x in inv.split('\n') if x] + modular_words
        incrementers = [x.strip() for x in inc.split('\n') if x]
        stl_ = [x for x in self.logic.get_stopwords() if
                x not in list(map(str, ['not', 'hope', 'need', 'would']))]
        phrase_dict = defaultdict(list)
        for index, comm in enumerate(unique_comments):
            comm = ' '.join(''.join(char for char in word if ord(char) < 128)
                            for word in comm.split())
            comm = comm.lower()
            for x in comm.split('.'):
                thanks_len = [i for i, s in enumerate(
                    x.lower().split()) if 'thank' in s]
                if thanks_len and 0 < len(x.split()) - thanks_len[-1] < 5:
                    x = ' '.join(x.split()[:thanks_len[-1]])
                else:
                    pass
                if x:
                    pos = [posvar for posvar in nltk.pos_tag(nltk.word_tokenize(x.lower()))
                        if posvar[0].lower() not in stl_]
                    ind = [(xvar, i) for i, xvar in enumerate(pos)]
                    ind = [lv for lv in [(indx, indy) if 'VB' not in indx[1] else (indx, indy) if len(indx[0]) > 3
                                        else '' for indx, indy in ind] if lv != '']
                    l = []
                    if ind:
                        nouns = [
                            arg_indvar for arg_indvar in ind if 'NN' in arg_indvar[0][1]]
                        # print((list(ind)))
                        adjs = [arg_indvar5 for arg_indvar5 in ind if
                                'VB' in arg_indvar5[0][1] or 'JJ' in arg_indvar5[0][1]]
                        advs = [
                            arg_indvar6 for arg_indvar6 in ind if arg_indvar6[0][0] in incrementers]
                        negators = [
                            arg_indvar7 for arg_indvar7 in ind if arg_indvar7[0][0] in negated_l]
                        if adjs:
                            for adjective in adjs:
                                adj_score = self.get_adj_score(adjective[0], pos_neg_score, positive_scores, negative_scores)
                                if nouns:
                                    # print('adjective[1]')
                                    # print(adjective[1])

                                    closest_noun = min(nouns,
                                                    key=lambda argvar_indvar2: abs(argvar_indvar2[1] - adjective[1]))
                                    # print('closest_noun')
                                    # print(closest_noun)

                                    if closest_noun[1] < adjective[1]:
                                        # Dont take words only since POS tag is needed
                                        sentiment_phrase = [
                                            x[0] for x in ind[closest_noun[1]:adjective[1] + 1]]
                                    else:
                                        sentiment_phrase = [
                                            x[0] for x in ind[adjective[1]:closest_noun[1] + 1]]
                                else:
                                    sentiment_phrase = x
                                if advs:
                                    closest_adverbs = [argvar_indvar3 for argvar_indvar3 in advs if
                                                    abs(argvar_indvar3[1] - adjective[1]) < 4]
                                    if closest_adverbs:
                                        nearest_adverb = min(closest_adverbs,
                                                            key=lambda argvar_indvar: adjective[1] - argvar_indvar[1])
                                        if nearest_adverb[0][0] in incrementers:
                                            adj_score = adj_score * 2

                                if negators:
                                    closest_negators = [argvar_indvar4 for argvar_indvar4 in negators if
                                                        abs(argvar_indvar4[1] - adjective[1]) < 4]
                                    if closest_negators:
                                        nearest_negator = min(closest_negators,
                                                            key=lambda argvar_indvar1: adjective[1] - argvar_indvar1[1])
                                        if nearest_negator[0][0] in negated_l:
                                            adj_score = adj_score * -1
                                l.append((sentiment_phrase, adj_score, adjective))
                            newl = {}
                            for avar in l:
                                if avar[2] not in newl and avar[1] != 0:
                                    newl[avar[2]] = (' '.join(avv[0]
                                                            for avv in avar[0]), avar[1])
                            phrase_dict[index].append(list(newl.values()))
                        else:
                            l.append(
                                (' '.join(t1[0] for t1 in [indtvar[0] for indtvar in ind]), 0))
                            phrase_dict[index].append(l)
                    else:
                        phrase_dict[index].append(l)
                else:
                    phrase_dict[index].append([])
        return phrase_dict

    def Build(self):
        """ 
            Build Method create and return new data frame
            Return
            ----------
            Dataframe 
        """
        phrase_dict = self.get_phrases_scores(self.data)
        unique_comments = self.data[Config.SurveyColumnName]
        phrasedictvals = {}
        for key, val in enumerate(unique_comments):
            phrasedictvals[key] = [xx for x in phrase_dict[key] for xx in x]

        total_scores = [sum(xx[1] for xx in x)
                        for x in phrasedictvals.values()]

        self.data[Config.SentimentColumnName] = total_scores
        return self.data
