# Word Sense Disambiguation

> Mathilde Tillman Hegdal | 91258 Natural Language Processing | February 2022
> <br />
> Word Sense Disambiguation (WSD) is the task of determining which sense of a word is being used in a context. It is one of the most important open problems in the field of Natural Language Processing and has numerous applications, from machine translation to information retrieval systems.
> <br />
> <br />
> <br />

**Methods**

In this project, three algorithms for solving WSD were implemented. The first is the **Simplified Lesk algorithm**, a knowledge-based algorithm that chooses the sense that has the most overlap between the target word's neighborhood and its definition and usage examples in the dictionary. The second approach is a supervised **Naive Bayes classifier** that predicts the sense by utilizing Bayesian inference and a simplified conditional independence assumption. It uses maximum likelihood estimation and add-one Laplace smoothing. Finally, a simple approach that chooses the **Most Frequent Sense** was implemented.
<br />
<br />

**Setting**

The setting for these experiments followed the standard benchmark for English word sense disambiguation. Princeton WordNet 3.0 was adopted as the sense inventory. The supervised system was trained using SemCor, and the testing was performed on the evaluation suite of Raganato et al.: SemEval-2007 Task 17, SemEval-2013 Task 12, SemEval-2015 Task 13, Senseval-2, and Senseval-3.
<br />
<br />

**Evaluation**

The performance of the three algorithms was measured in F1-Score by running the Scorer.java script provided by Ragnato et al.
To use the scorer, you first need to compile:

    $ javac Scorer.java

Then, evaluate your system by typing the following command: java Scorer [gold-standard] [system-output]

Example of usage:

    $ java Scorer semeval2013/semeval2013.gold.key.txt semeval2013/output.key

The results are displayed in table below.

| System              | ALL  | S2   | S3   | S7   | S13  | S15  |
| ------------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| Simplified Lesk     | 38.9 | 39.4 | 37.1 | 25.3 | 43.8 | 38.9 |
| Naive Bayes         | 53.3 | 53.4 | 56.2 | 52.3 | 49.2 | 54.8 |
| Most Frequent Sense | 40.7 | 39.6 | 38.8 | 27.7 | 49.5 | 38.1 |
