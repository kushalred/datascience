import logging  # to import log levels
from pathlib import Path
import os


class Config(object):
    # specifying filename to store logs
    LOG_FILENAME = str(Path(__file__).parent) + os.sep + \
        "logs" + os.sep + "process.log"
    # defining custom log format
    LOG_FORMAT = "%(asctime)s:%(msecs)03d %(levelname)s %(filename)s:%(lineno)d %(message)s"
    # defining the date format for logger
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    # defining log levels
    LOG_LEVEL = logging.DEBUG

    # Output folder name
    OUTPUT_LOCATION = str(Path(__file__).parent) + os.sep + "output"
    # topic file Location
    TOPIC_FILE_PATH = str(Path(__file__).parent) + \
        os.sep + "output" + os.sep + "topics1.csv"
    # Topic Detil output file name
    TOPIC_DETAIL_FILE_PATH = str(
        Path(__file__).parent) + os.sep + "output" + os.sep + "topic_detail.csv"
    # Stopword file
    STOPWORDS = str(Path(__file__).parent) + os.sep + \
        "corpus" + os.sep + "stopwords.txt"
    # Topis Keywords
    TOPIC_KEYOWRD = str(Path(__file__).parent) + os.sep + \
        "corpus" + os.sep + "topic_keyword.csv"
    # Survey Column name
    SurveyColumnName = 'feedback__c'
    # Action Need Column Name
    ActionColumnName = "Action Needed"
    # Sentiment Score Column Name
    SentimentColumnName = "Sentiment_Score"
    # Topic Column Name
    TopicColumnName = "Topic"
    # Detail Table Column Name
    TOPIC_DETAIL_COLUMN = list(["Id", "Name", "Count"])

    # Sentiment Corpus files
    # Sentiment Traning set
    TRANINGSET = str(Path(__file__).parent) + os.sep + \
        "corpus" + os.sep + "training_set.txt"
    # Positive words
    POSITIVEWORDS = str(Path(__file__).parent) + os.sep + \
        "corpus" + os.sep + "new_positive.txt"
    # Negitive words
    NEGITIVEWORDS = str(Path(__file__).parent) + os.sep + \
        "corpus" + os.sep + "new_negative.txt"
    # INC words
    INCWORDS = str(Path(__file__).parent) + os.sep + \
        "corpus" + os.sep + "inc.txt"
    # INV Words
    INVWORDS = str(Path(__file__).parent) + os.sep + \
        "corpus" + os.sep + "inv.txt"
    # AFINN Words
    AFINNWORDS = str(Path(__file__).parent) + os.sep + \
        "corpus" + os.sep + "AFINN-111.txt"
