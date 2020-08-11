import nltk
import pandas as pd
from fuzzywuzzy import process

from custom_logic import CustomLogic
from config import Config

def TopicScore(text, words):
        """ 
            Get Topic Score based on Text and Topic words
            Parameters
            ----------
            text: string
            words: list

            Return
            ----------
            1 -> Topic avaliable
            0 -> Not 
        """
        score = process.extractOne(text, words)[1]
        if score > 55:
            return 1
        else:
            return 0


class TopicAnalyser():
    """ 
        Topic Analyser build topic based on topic_keyword.csv file and return data frame
    """
    def __init__(self, data_file):
        self.data = data_file
        self.logic = CustomLogic()
        self.topics = self.GetTopicWords()

    
    def GetTopicWords(self):
        """ 
            Get list of Topic from topic CSV file 
            Return
            ----------
            Dict of Topic keyword
        """
        topic = pd.read_csv(Config.TOPIC_KEYOWRD, encoding='iso-8859-1')
        topic_dict = dict()
        # Loop list of topic
        for column in topic.columns:
            topic_dict[column] = list(topic[topic[column].notnull()][column])

        return topic_dict    

    
    def Build(self):
        """ 
          Build Table with Topic in Master dataset
          Return
          --------------
          Topic -> Add new topic column 
          Topic_Detail -> Detail of Topic in rows
        """
        topic_dict = self.GetTopicWords()
        # Create Topic column and add to dataframe
        for key, value in topic_dict.items():
            self.data[key] = self.data[Config.SurveyColumnName].apply(TopicScore, args=[value])

        # Create new Detail Table
        df_detail = pd.DataFrame(columns=Config.TOPIC_DETAIL_COLUMN)
        # List Topic Name
        topic_name = []
        sep = '.'
        for index, row in self.data.iterrows():
            _names = []
            # Loop list of topic
            for key, value in topic_dict.items():
                # Create new series of rows for detail table
                detail_row = pd.Series([row['id'],key,row[key]], index = df_detail.columns)
                df_detail = df_detail.append(detail_row, ignore_index=True)
                # Check dataframe of that topic is 1 if yes the add the name
                if row[key] == 1:
                    _names.append(key)
            
            # Concate the topic string name
            topic_name.append(sep.join(_names))

        # Add to new column to matched topic
        self.data[Config.TopicColumnName] = topic_name

        return self.data, df_detail

    
