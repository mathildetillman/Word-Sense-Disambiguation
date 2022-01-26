import pandas as pd
import numpy as np
import re
from operator import itemgetter


class NaiveBayesWSD:

    def __init__(self):
        self.train = None                   # Training dataset
        self.semeval2007 = None             # Test dataset
        self.semeval2013 = None             # Test dataset
        self.semeval2015 = None             # Test dataset
        self.senseval2 = None               # Test dataset
        self.senseval3 = None               # Test dataset

        self.classes = []                   # List of sense classes in training dataset
        self.classCounts = None             # Series of with index: class, value: total count in training dataset
        self.wordClasses = None             # Series with index: word, value: set of possible classes in training dataset

        self.vocabulary = []                # List of vocabulary in training dataset
        self.nVoc = 0                       # Length of vocabulary in training dataset
        self.aprioriProbabilites = []       # List of the apriori probabilities for the sense classes in the training dataset

        self.alpha = 1                      # Smoothing parameter


    def loadData(self):
        """
        Load the training and test datasets, and remove rows with NaN values.
        """ 

        # Load the training dataset
        train = pd.read_csv("https://raw.githubusercontent.com/mathith/Word-Sense-Disambiguation-Data/main/semcor.csv")
        self.train = train.dropna().reset_index(drop = True)

        # Load the test datasets
        semeval2007 = pd.read_csv("https://raw.githubusercontent.com/mathith/Word-Sense-Disambiguation-Data/main/semeval2007.csv")
        semeval2013 = pd.read_csv("https://raw.githubusercontent.com/mathith/Word-Sense-Disambiguation-Data/main/semeval2013.csv")
        semeval2015 = pd.read_csv("https://raw.githubusercontent.com/mathith/Word-Sense-Disambiguation-Data/main/semeval2015.csv")
        senseval2 = pd.read_csv("https://raw.githubusercontent.com/mathith/Word-Sense-Disambiguation-Data/main/senseval2.csv")
        senseval3 = pd.read_csv("https://raw.githubusercontent.com/mathith/Word-Sense-Disambiguation-Data/main/senseval3.csv")

        self.semeval2007 = semeval2007.dropna().reset_index(drop = True) 
        self.semeval2013 = semeval2013.dropna().reset_index(drop = True)
        self.semeval2015 = semeval2015.dropna().reset_index(drop = True) 
        self.senseval2 = senseval2.dropna().reset_index(drop = True) 
        self.senseval3 = senseval3.dropna().reset_index(drop = True) 


    def trainModel(self):
        """
        Trains the model by finding and calculating all the variables needed for the classification.
        """ 

        self.classes = list(self.train["sense_class"].unique())
        self.classCounts = self.train["sense_class"].value_counts()
        self.wordClasses = self.train.groupby("target_word")["sense_class"].agg(set)

        # VOCABULARY
        vocabulary = ""
        for row in self.train.iterrows():
            vocabulary += str(row[1]["context_string"]) + " "
        vocabulary = vocabulary.strip().split(" ")
        vocabulary = list(set(vocabulary))

        self.vocabulary = vocabulary
        self.nVoc = len(vocabulary)

        # APRIORI PROBABILITES
        num_rows = self.train.shape[0]
        for class_name in self.classes:
            self.aprioriProbabilites.append(np.log(self.classCounts[class_name] / num_rows))

        # PREPARE TRAINING DATASET
        # Group by sense_class and join all context strings
        self.train = self.train.drop(["target_word"], axis = 1).groupby(['sense_class'])['context_string'].apply(" ".join).reset_index()
        
        # Get total word count for each class
        self.train["word_count"] = self.train["context_string"].apply(lambda x: len(x))
        self.train = self.train.set_index("sense_class")
    

    def getLikelihood(self, class_name, context_word):
        """
        Calculates the log likelihood of a word given the class 

        :param class_name: sense class under investigation
        :param context_word: word in context under investigation
        :return: the log likelihood of a word given a class
        """ 

        # Access all words in all contexts for a class that appeared in the training data
        all_words_given_class = self.train.at[class_name, "context_string"]

        # Get frequency of word given class
        word_count_given_class = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(context_word), all_words_given_class))

        # Calulate log likelihood with smoothing
        likelihood = np.log((word_count_given_class + self.alpha)/(self.train.at[class_name, "word_count"] + self.alpha * self.nVoc))

        return likelihood


    def classify(self, row): 
        """
        Classifies a word with a sense class. 

        :param row: contains <id, target_word, context_string>, 
                    where target_word is the word to be classified and context_string is the context
        :return: the most likely class of the word
        """ 

        word = row.target_word
        context = row.context_string.split()       
        scores = []     # List of the likelihood of the possible senses for the word in context

        # If word to be classified does not exist in training data
        if (not (word in self.wordClasses.index)):
            return "None"

        # Else if only one possible class
        if (len(self.wordClasses[word]) == 1):
            predicted = next(iter(self.wordClasses[word]))
            return predicted

        # Else choose class with highest probability given context
        # Perform calculations in log space to avoid underflow and increase speed
        for class_name in self.wordClasses[word]:

            # Get apriori probability of class
            class_index = self.classes.index(class_name)
            p_class_given_message = self.aprioriProbabilites[class_index]

            # Get likelihood of word in context given class
            for context_word in context:
                if context_word in self.vocabulary:
                    p_class_given_message += self.getLikelihood(class_name, context_word)                
            scores.append((class_name, p_class_given_message))
        return max(scores, key = itemgetter(1))[0]
                

    def runClassification(self):
        """
        Classify all the words in the test datasets and save results to file.
        """ 

        self.semeval2007["predicted"] = self.semeval2007.apply(lambda row: self.classify(row), axis=1)
        self.semeval2007 = self.semeval2007.drop(["target_word", "context_string"], axis = 1)
        self.semeval2007.to_csv("results/naiveBayes/nb_semeval2007_predicted.txt", sep=' ', header = False, index = False)

        self.semeval2013["predicted"] = self.semeval2013.apply(lambda row: self.classify(row), axis=1)
        self.semeval2013 = self.semeval2013.drop(["target_word", "context_string"], axis = 1)
        self.semeval2013.to_csv("results/naiveBayes/nb_semeval2013_predicted.txt", sep=' ', header = False, index = False)

        self.semeval2015["predicted"] = self.semeval2015.apply(lambda row: self.classify(row), axis=1)
        self.semeval2015 = self.semeval2015.drop(["target_word", "context_string"], axis = 1)
        self.semeval2015.to_csv("results/naiveBayes/nb_semeval2015_predicted.txt", sep=' ', header = False, index = False)

        self.senseval2["predicted"] = self.senseval2.apply(lambda row: self.classify(row), axis=1)
        self.senseval2 = self.senseval2.drop(["target_word", "context_string"], axis = 1)
        self.senseval2.to_csv("results/naiveBayes/nb_senseval2_predicted.txt", sep=' ', header = False, index = False)

        self.senseval3["predicted"] = self.senseval3.apply(lambda row: self.classify(row), axis=1)
        self.senseval3 = self.senseval3.drop(["target_word", "context_string"], axis = 1)
        self.senseval3.to_csv("results/naiveBayes/nb_senseval3_predicted.txt", sep=' ', header = False, index = False)




