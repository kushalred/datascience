from config import Config
from nltk.corpus import stopwords
import re
import string

class CustomLogic():
    """ 
        Custom Logic used on all class to aviod repeat of code
        Methods
        ---------
        get_custom_stopwords()
        get_stopwords()
    """

    def get_custom_stopwords(self):
        """ 
            Get list of all stopword added in corpus/stopword.txt file
            Returns
            ----------
            list
        """
        f = open(Config.STOPWORDS, "r")
        words = []
        for x in f:        
            words.append(str(re.findall(r'\w+', x)[0]))
        
        return words

    def get_stopwords(self):
        """ 
            Get all stop words of nltk stopword + string.punctuation + custom stopword
            Returns
            ----------
            list
        """
        custom_stopwords = self.get_custom_stopwords()
        stop_words = stopwords.words('english') + list(string.punctuation) + list(custom_stopwords)
        return stop_words