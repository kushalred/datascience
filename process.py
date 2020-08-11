import sys
import pandas as pd
import os
import shutil
import logging

from custom_logic import CustomLogic
from action_words import ActionWords
from topic import TopicAnalyser
from sentiment import Sentiment
from config import Config

### Log Error
import logging

# Configuring Logs
root_logger = logging.getLogger()
root_logger.setLevel(Config.LOG_LEVEL)
handler = logging.FileHandler(Config.LOG_FILENAME, 'a+', 'utf-8')
handler.setFormatter(logging.Formatter(fmt=Config.LOG_FORMAT, datefmt=Config.LOG_DATE_FORMAT))
root_logger.addHandler(handler)


def clean(filename):
    """ Clean Data Frame file before process 
        Parameters
        ----------
        filename : str
        The file location of the spreadsheet 
    """
    df = pd.read_csv(filename)
    file_process = df.drop_duplicates([Config.SurveyColumnName]).reset_index(drop = True)
    file_process = file_process.dropna(subset = [Config.SurveyColumnName]).reset_index(drop = True)
    return file_process

def clear_output():
    """ Clear Output dir before add new output files """
    shutil.rmtree(Config.OUTPUT_LOCATION)
    os.makedirs(Config.OUTPUT_LOCATION)

def Process(filename):
    """ 
        Process  all output like action need, sentiment score and Topics. Save the output in two files
        Topic.csv and Topic_detail.csv
    
    """
    try:
        root_logger.info('Process Started %s' % filename) 
        # delete old files
        clear_output()

        # clean data set 
        data = clean(filename)

        # Load Action data
        action = ActionWords(data)        
        action_data = action.action_needed()

        # Load Sentiment
        sent = Sentiment(action_data)
        sentiment_data = sent.Build()

        # Load Topic
        topic = TopicAnalyser(sentiment_data)        
        topic_data, topic_detail_data = topic.Build()

        # Save output file
        topic_data.to_csv(Config.TOPIC_FILE_PATH, index = False, encoding = 'utf-8')
        topic_detail_data.to_csv(Config.TOPIC_DETAIL_FILE_PATH, index = False, encoding = 'utf-8')

        root_logger.info('Process Completed %s' % filename) 
    except Exception as e:        
        root_logger.error('Error %s' % e) 
        
    
if __name__== '__main__':
    input_file = sys.argv[1]
    #print(input_file)
    Process(input_file)