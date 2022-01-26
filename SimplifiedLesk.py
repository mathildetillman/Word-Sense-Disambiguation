import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

class SimplifiedLesk:

    def loadData(self):
        """
        Load the test datasets, and remove rows with NaN values.
        """ 

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


    def preprocess(self, text):
        """
        Preprocess the text by tokenization, lemmatization, removing punctuation, stopwords, forbidden words and numbers, and converting to lowercase.

        :param text: the string to perform the processing on
        :return: processed string as a list of words
        """ 

        forbidden_words =  ["&apos;", "``", "''", "'", "`", "'s"] 

        # Tokenize and convert to lowercase
        tokens = word_tokenize(text.lower())

        # Remove stopwords, punctuation, forbidden words and numbers
        tokens = [word for word in tokens if not word in stopwords.words('english') and not word in punctuation and not word in forbidden_words and not word.isnumeric()]

        # Remove duplicates
        tokens = list(set(tokens))

        return tokens


    def computeOverlap(signature, context):
        """
        Count words that occur in both lists

        :param signature: list of words of the glosses and examples of a synset
        :param context: list of words in the context
        :return: number of words in common
        """ 
        overlap = 0
        for token in context:
            if token in signature:
                overlap += 1
        return overlap


    
    def classify(self, row):
        """
        Classifies a word with a sense class. 

        :param row: contains <id, target_word, context_string>, 
                    where target_word is the word to be classified and context_string is the context
        :return: the predicted sense class of the word
        """ 

        word = row.target_word
        context =  word_tokenize(row.context_string)
        synsets = wn.synsets(word)

        if len(synsets) > 0:
            best_sense = synsets[0]
            max_overlap = 0

            for synset in synsets:

                # Add definition and examples to signature
                signature = synset.definition()
                for example in synset.examples():
                    signature += " " + example
                signature = self.preprocess(signature)

                # Find overlap between signature and context
                overlap = self.computeOverlap(signature, context)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_sense = synset
            return best_sense
        return "None"


    def runClassification(self):
        """
        Classify all the words in the test datasets and save results to file.
        """ 

        self.semeval2007["predicted"] = self.semeval2007.apply(lambda row: self.classify(row), axis=1)
        self.semeval2007 = self.semeval2007.drop(["target_word", "context_string"], axis = 1)
        self.semeval2007.to_csv("results/lesk/lesk_semeval2007_predicted.txt", sep=' ', header = False, index = False)

        self.semeval2013["predicted"] = self.semeval2013.apply(lambda row: self.classify(row), axis=1)
        self.semeval2013 = self.semeval2013.drop(["target_word", "context_string"], axis = 1)
        self.semeval2013.to_csv("results/lesk/lesk_semeval2013_predicted.txt", sep=' ', header = False, index = False)

        self.semeval2015["predicted"] = self.semeval2015.apply(lambda row: self.classify(row), axis=1)
        self.semeval2015 = self.semeval2015.drop(["target_word", "context_string"], axis = 1)
        self.semeval2015.to_csv("results/lesk/lesk_semeval2015_predicted.txt", sep=' ', header = False, index = False)

        self.senseval2["predicted"] = self.senseval2.apply(lambda row: self.classify(row), axis=1)
        self.senseval2 = self.senseval2.drop(["target_word", "context_string"], axis = 1)
        self.senseval2.to_csv("results/lesk/lesk_senseval2_predicted.txt", sep=' ', header = False, index = False)

        self.senseval3["predicted"] = self.senseval3.apply(lambda row: self.classify(row), axis=1)
        self.senseval3 = self.senseval3.drop(["target_word", "context_string"], axis = 1)
        self.senseval3.to_csv("results/lesk/lesk_senseval3_predicted.txt", sep=' ', header = False, index = False)