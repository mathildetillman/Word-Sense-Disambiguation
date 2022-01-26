import pandas as pd
import xml.etree.ElementTree as ET
import copy
from string import punctuation
import spacy


def preprocess(name, goldKeyFilePath, xmlFilePath, isTrainingSet):
    """
    Parse XML file and preprocess dataset:
    Perform lemmatization, remove punctuation, stopwords and forbidden words, convert to lowercase and remove numbers.
    If the dataset is going to be used for training - add the senses

    :param 
    """ 

    # List of rows to insert into dataframe
    rows_list = []

    # Load gold key file with senses if dataset is training set
    if (isTrainingSet):
        senses = pd.read_table(goldKeyFilePath, header = None, sep=' ', names=['id', 'sense_class'])
    
    # Load XML file
    xmlTree = ET.parse(xmlFilePath)
    corpus = xmlTree.getroot()

    # Want to clean data: remove stopwords and other non-words
    sp = spacy.load('en_core_web_sm')
    stopwords = sp.Defaults.stop_words
    forbidden_words =  ["&apos;", "``", "''", "'", "`", "'s"] 

    for text in corpus:
        for sentence in text:
            context = []
            instances = []
            for word in sentence:
                context.append(word.attrib["lemma"])
                if (word.tag == "instance"):
                    instances.append((word.attrib["id"], word.attrib["lemma"]))
            
            # Ensure lower-case an remove punctuation, numbers, non-words and stopwords
            context = [word.lower() for word in context if not word in punctuation and not word.isnumeric() and word not in forbidden_words and word not in stopwords]
            for instance in instances:

                # Remove target_word from context if it exists
                contextCopy = copy.deepcopy(context)
                target_word = instance[1]
                if(target_word in contextCopy):
                    index = contextCopy.index(target_word)
                    contextCopy.pop(index)
                contextCopy = " ".join(contextCopy)

                row_dict = {"id": instance[0], "target_word": instance[1], "context_string": contextCopy}
                rows_list.append(row_dict)


    cleaned_data = pd.DataFrame(rows_list) 

    # Join with the gold key file if the dataset is the training set
    if (isTrainingSet):
        cleaned_data = cleaned_data.set_index('id').join(senses.set_index('id'))

    cleaned_data.to_csv("data/cleaned/" + name + ".csv", index = False)


