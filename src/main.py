import sys
from util import Util

def main():

    util = Util()
    print('------------------------------------------------------')
    print('------- WELCOME TO PERCEPTRON EMAIL CLASSIFIER -------')
    print('------------------------------------------------------')
    userInput = input('Please enter the amount of iterations or hit enter for default, (default: 1000):')
    if(len(userInput) != 0 and userInput != " " and userInput != ""):
        iterations = userInput
        print('Running on '+iterations+'iterations!')
    else:
        iterations = '1000'
        print('Using default of 1000!')

    userInput = input('Now, enter the learning rate or hit enter for default, (default: .001):')
    if(len(userInput) != 0 and userInput != " " and userInput != ""):
        learning_constant = userInput
        print('Running on '+learning_constant+'learning constant!')
    else:
        learning_constant = '.001'
        print('Using default of .001!')
    
    print('Running...')
        
    # Create dictionaries and lists needed
    training_set = {}
    test_set = {}
    filtered_training_set = {}
    filtered_test_set = {}

    # Getting stop words to use as filter
    stop_words = util.setStopWords('../data/stop_words.txt')

    # Spam and Ham classes that will be used to divide
    # ham = 0 to point what is not a spam, spam = 1 to point what is a spam
    classes = ["ham", "spam"]

    # Creating the needed datasets from the available files so that we can run a train and test process
    # Dictionaries containing the text, word frequencies, and true/learned classifications
    util.makeDataSet(training_set, "../data/train/spam", classes[1])
    util.makeDataSet(training_set, "../data/train/ham", classes[0])
    util.makeDataSet(test_set, "../data/test/spam", classes[1])
    util.makeDataSet(test_set, "../data/test/ham", classes[0])

    # Filtered train and test sets filtered so that the stop words are removed
    filtered_training_set = util.removeStopWords(stop_words, training_set)
    filtered_test_set = util.removeStopWords(stop_words, test_set)

    # Extract training vocabulary set raw and filtered 
    training_set_vocab = util.extractVocab(training_set)
    filtered_training_set_vocab = util.extractVocab(filtered_training_set)

    # Setting up the weights as dictionaries
    # w0 initial 1.0, others initially 1.0. token : weight value
    weights = {'weight_zero': 1.0}
    filtered_weights = {'weight_zero': 1.0}
    for i in training_set_vocab:
        weights[i] = 0.0
    for i in filtered_training_set_vocab:
        filtered_weights[i] = 0.0

    # Getting the algorithm to learn the set of weights by using the training_set and filtered_training_set
    util.learnWeights(weights, learning_constant, training_set, iterations, classes)
    util.learnWeights(filtered_weights, learning_constant, filtered_training_set, iterations, classes)

    # Applying the algorithm
    # We now run the perceptron algorithm on the test set and report accuracy
    num_correct_guesses = 0
    for i in test_set:
        guess = util.apply(weights, classes, test_set[i])
        if guess == 1:
            test_set[i].setLearnedClass(classes[1])
            if test_set[i].getTrueClass() == test_set[i].getLearnedClass():
                num_correct_guesses += 1
        if guess == 0:
            test_set[i].setLearnedClass(classes[0])
            if test_set[i].getTrueClass() == test_set[i].getLearnedClass():
                num_correct_guesses += 1

    # Applying the algorithm
    # We now run the perceptron algorithm on the test set and report accuracy but now without the stop words with the filtered sets
    filt_num_correct_guesses = 0
    for i in filtered_test_set:
        guess = util.apply(filtered_weights, classes, filtered_test_set[i])
        if guess == 1:
            filtered_test_set[i].setLearnedClass(classes[1])
            if filtered_test_set[i].getTrueClass() == filtered_test_set[i].getLearnedClass():
                filt_num_correct_guesses += 1
        if guess == 0:
            filtered_test_set[i].setLearnedClass(classes[0])
            if filtered_test_set[i].getTrueClass() == filtered_test_set[i].getLearnedClass():
                filt_num_correct_guesses += 1

    # Printing out the report
    print('------------------------------------------------------------')
    print('-------------------------- REPORT --------------------------')
    print('------------------------------------------------------------')
    print("Learning constant: %.4f" % float(learning_constant))
    print("Number of iterations: %d" % int(iterations))
    print("Emails classified correctly: %d/%d" % (num_correct_guesses, len(test_set)))
    print("Accuracy: %.4f%%" % (float(num_correct_guesses) / float(len(test_set)) * 100.0))
    print("Filtered emails classified correctly: %d/%d" % (filt_num_correct_guesses, len(filtered_test_set)))
    print("Filtered accuracy: %.4f%%" % (float(filt_num_correct_guesses) / float(len(filtered_test_set)) * 100.0))


if __name__ == '__main__':
    main()