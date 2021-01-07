import collections
import re
import copy
import os
from email_classifier import EmailClassifier
import matplotlib.pyplot as plt
import numpy as np

class Util:

    # counts frequency of each word in the text files and order of sequence doesn't matter
    def bagOfWords(self, text):
        bagsofwords = collections.Counter(re.findall(r'\w+', text))
        return dict(bagsofwords)

    # Read all text files in given directory and construct the data set
    # the directory path should just be like "train/ham" for example
    # storage is the dictionary to store the email in
    # True class is the true classification of the email (spam or ham)
    def makeDataSet(self, storage_dict, directory, true_class):
        for dir_entry in os.listdir(directory):
            dir_entry_path = os.path.join(directory, dir_entry)
            if os.path.isfile(dir_entry_path):
                with open(dir_entry_path, encoding="latin-1") as text_file:
                    # stores dictionary of dictionary of dictionary as explained above in the initialization
                    text = text_file.read()
                    storage_dict.update({dir_entry_path: EmailClassifier(text, self.bagOfWords(text), true_class)})

    # Set the stop words
    def setStopWords(self, stop_word_text_file):
        stops = []
        with open(stop_word_text_file, encoding="latin-1") as txt:
            stops = (txt.read().splitlines())
        return stops

    # Remove stop words from data set and store in dictionary
    def removeStopWords(self, stops, data_set):
        filtered_data_set = copy.deepcopy(data_set)
        for i in stops:
            for j in filtered_data_set:
                if i in filtered_data_set[j].getWordFreqs():
                    del filtered_data_set[j].getWordFreqs()[i]
        return filtered_data_set

    # Extracts the vocabulary of all the text in a data set
    def extractVocab(self, data_set):
        v = []
        for i in data_set:
            for j in data_set[i].getWordFreqs():
                if j not in v:
                    v.append(j)
        return v

    # learns weights using the perceptron training rule
    def learnWeights(self, weights, learning_constant, training_set, num_iterations, classes):
        # Adjust weights num_iterations times
        for i in num_iterations:
            # Go through all training instances and update weights
            for d in training_set:
                # Used to get the current perceptron's output. If > 0, then spam, else output ham.
                weight_sum = weights['weight_zero']
                for f in training_set[d].getWordFreqs():
                    if f not in weights:
                        weights[f] = 0.0
                    weight_sum += weights[f] * training_set[d].getWordFreqs()[f]
                perceptron_output = 0.0
                if weight_sum > 0:
                    perceptron_output = 1.0
                target_value = 0.0
                if training_set[d].getTrueClass() == classes[1]:
                    target_value = 1.0
                # Update all weights that are relevant to the instance at hand
                for w in training_set[d].getWordFreqs():
                    weights[w] += float(learning_constant) * float((target_value - perceptron_output)) * \
                                float(training_set[d].getWordFreqs()[w])


    # applies the algorithm to test accuracy on the test set. Returns the perceptron output
    def apply(self, weights, classes, instance):
        weight_sum = weights['weight_zero']
        for i in instance.getWordFreqs():
            if i not in weights:
                weights[i] = 0.0
            weight_sum += weights[i] * instance.getWordFreqs()[i]
        if weight_sum > 0:
            # return is spam
            return 1
        else:
            # return is ham
            return 0