from preprocess import preprocess
from SimplifiedLesk import SimplifiedLesk
from NaiveBayesWSD import NaiveBayesWSD
from MostFrequentSense import MostFrequentSense

DATA = [
            {   
                "name": "semcor",
                "goldKeyFp": "data\original\semcor\semcor.gold.key.clean.txt",
                "xmlFp": "data\original\semcor\semcor.data.xml",
                "isTrainingSet": True
            },
            {
                "name": "all",
                "goldKeyFp": "data/original/allTest/ALL.gold.key.txt",
                "xmlFp": "data/original/allTest/ALL.data.xml",
                "isTrainingSet": False
            },
            {
                "name": "semeval2007",
                "goldKeyFp": "data\original\semeval2007\semeval2007.gold.key.txt",
                "xmlFp": "data\original\semeval2007\semeval2007.data.xml",
                "isTrainingSet": False
            },
            {
                "name": "semeval2013",
                "goldKeyFp": "data\original\semeval2013\semeval2013.gold.key.txt",
                "xmlFp": "data\original\semeval2013\semeval2013.data.xml",
                "isTrainingSet": False
            },
            {
                "name": "semeval2015",
                "goldKeyFp": "data\original\semeval2015\semeval2015.gold.key.txt",
                "xmlFp": "data\original\semeval2015\semeval2015.data.xml",
                "isTrainingSet": False
            }, {
                "name": "senseval2",
                "goldKeyFp": "data\original\senseval2\senseval2.gold.key.txt",
                "xmlFp": "data\original\senseval2\senseval2.data.xml",
                "isTrainingSet": False
            },
            {
                "name": "senseval3",
                "goldKeyFp": "data\original\senseval3\senseval3.gold.key.txt",
                "xmlFp": "data\original\senseval3\senseval3.data.xml",
                "isTrainingSet": False
            }
        ]

def main():


    # 1 - Preprocess the datasets
    for dataset in DATA:
        preprocess(dataset["name"], dataset["goldKeyFp"], dataset["xmlFp"], dataset["isTrainingSet"])

    # 2 - Perform word sense disambiguation using the simplified Lesk algorithm
    simplifiedLesk = SimplifiedLesk() 
    simplifiedLesk.loadData()
    simplifiedLesk.runClassification()


    # 3 - Perform word sense disambiguation using the Naive Bayes classifier
    naiveBayes = NaiveBayesWSD() 
    naiveBayes.loadData()
    naiveBayes.trainModel()
    naiveBayes.runClassification()

    # 4 - Perform word sense disambiguation using the most frequent sense
    mostFrequentSense = MostFrequentSense()
    mostFrequentSense.loadData()
    mostFrequentSense.runClassification()




if __name__ == '__main__':
    main()