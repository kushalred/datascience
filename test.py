from custom_logic import CustomLogic
from action_words import ActionWords
from topic import TopicAnalyser
from config import Config
from sentiment import Sentiment
from nltk.corpus import stopwords

import sys
import pandas as pd
import os
import shutil
#https://github.com/kushalred/kushal.git

def test_custom_logic():
    logic = CustomLogic()
    stopwords = logic.get_stopwords()
    print(stopwords)


def test_action_need(file):
    action = ActionWords(file)
    new_data = action.action_needed()
    #new_data.to_csv(Config.OUTPUT_LOCATION,index = False, encoding = 'utf-8')


def process(filename):
    df = pd.read_csv(filename)
    file_process = df.drop_duplicates(
        [Config.SurveyColumnName]).reset_index(drop=True)
    file_process = file_process.dropna(
        subset=[Config.SurveyColumnName]).reset_index(drop=True)
    return file_process


def test_gettopics(df):
    topic = TopicAnalyser(df)
    topic_data, topic_data_detail = topic.Build()
    topic_data.to_csv(Config.TOPIC_FILE_PATH, index=False, encoding='utf-8')
    topic_data_detail.to_csv(
        Config.TOPIC_DETAIL_FILE_PATH, index=False, encoding='utf-8')
    print('Topic:{}, Detail: {}'.format(
        topic_data.shape, topic_data_detail.shape))


def cleatoutput():
    shutil.rmtree(Config.OUTPUT_LOCATION)
    os.makedirs(Config.OUTPUT_LOCATION)


def test_sentiment(df):
    # cleatoutput()
    sent = Sentiment(df)
    new_data = sent.Build()
    new_data.to_csv(Config.TOPIC_FILE_PATH, index=False, encoding='utf-8')


if __name__ == '__main__':
    # test_custom_logic()
    # cleatoutput()
    input_file = r'C:\Users\Chaithanya Kushal\Desktop\TMFEED.csv'  # sys.argv[1]
    df = process(input_file)
    test_sentiment(df)
    # test_gettopics(df)
    #df = process(input_file)
    # test_action_need(df)