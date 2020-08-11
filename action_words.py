import nltk
import pandas as pd
from custom_logic import CustomLogic
from config import Config

class ActionWords():    
    """ 
        Action Word class find action based on feedback 

        Methods
        ------------
        action_needed()
    """

    def __init__(self, data_file):
        """ 
            Parameters
            ----------
            data_file : Pandas data frame file
        """
        self.data = data_file
        self.logic = CustomLogic()
        self.frustrationWord = "Frustration"
        self.callbackWord = "Call Back"
        self.naWord = 'NA'
        self.frustation = [('numerous', 'times'), ('many', 'times'), ('several', 'times'), 
                             ('third', 'time'), ('second', 'time'), ('fourth', 'time'), 
                             ('second', 'call'), ('third', 'call'), ('fourth', 'call')]
        self.callback = [('call', 'back'), ('call', 'me'), ('give', 'call'), ('me', 'call'), 
                         ('phone', 'number'), ('like', 'talk')]

    
    def action_needed(self):
        '''
        The function takes in a dataframe and checks if any action is needed 
        and adds a new column which denotes what action is needed based on the feedback

        Returns
        -------
        dataframe with action need column
        '''
        # Get list of stop words
        self.stopwords = self.logic.get_stopwords()
        st = [svar for svar in self.stopwords if svar not in list(map(str, ['not', 'me']))]
        callback_l = []
        for i, row in self.data.iterrows():
            tokens = [word.lower().strip() for word in row[Config.SurveyColumnName].split() if word.strip().lower() not in st]
            # training set for many calls from the user For frustation
            bigram_for_frustation = [comb for comb in nltk.bigrams(tokens) if
                                    comb in self.frustation]
            if [comb for comb in nltk.ngrams(tokens, 2) if comb in self.callback]:
                callback_l.append(self.callbackWord)
            elif bigram_for_frustation:
                # Get the index of the first word in these bigrams so that we can check the previous words
                first_word_ind = tokens.index(bigram_for_frustation[0][0])
                if tokens[first_word_ind + 1] == bigram_for_frustation[0][1] and \
                            'call' in tokens[first_word_ind - 2:first_word_ind + 2] or 'called' in tokens[
                            first_word_ind - 2:first_word_ind + 4]:
                    # Check if the user has talked about having made repeated calls
                    callback_l.append(self.frustrationWord)
                elif ('problem' in tokens and 'resol' in row[Config.SurveyColumnName]) \
                            or ('not' in tokens and 'resol' in row[Config.SurveyColumnName]) or \
                                ('many' in tokens and 'days' in tokens):
                    # Check for problem not resolved for a long time
                    callback_l.append(self.frustrationWord)
                elif 'wait' in row[Config.SurveyColumnName] or ('long' in tokens and 'time' in tokens):
                    # Check for customer waiting for a long time
                    callback_l.append(self.frustrationWord)
                else:
                    # If none of the above is satisfied, add a NA to the list
                    callback_l.append(self.naWord)
            else:
                callback_l.append(self.naWord)

        self.data[Config.ActionColumnName] = callback_l
        return self.data
