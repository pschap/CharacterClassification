import pandas as pd
from collections import Counter

# Counts the total number of instances of each decision class in a given data frame.
#
# Parameters:
# df - the data frame
def get_number_of_instances(df):
    
    number_of_instances = Counter()
    category_index = 1
    
    for row in df.itertuples():
        category = row[category_index]
        number_of_instances[category] += 1
        
    return number_of_instances
    
# Given a dictionary that contains the number of instances of each individual decision class
# in a data set, calculates and returns the total number of data instances present throughout
# the entire data set.
#
# Parameters:
# instances - dictionary object that stores number of instances of each decision class in the data set
def total_number_of_instances(instances):

    count = 0
    
    for item in instances.keys():
        count += instances[item]
        
    return count
  
# Given dictionary that contains the number of instances of each individual decision class
# in a data set, along with the total number of data instances throughout the entire data set,
# calculates the probability of each decision class appearing throughout the data set and returns
# the results as a dictionary object.
#
# Parameters:
# instances - dictionary object that stores the number of instances of each decision class in the data set
# total_instances - the total number of data instances throughout the entire data set
def find_decision_class_probabilities(instances, total_instances):
    class_probabilities = {}
    
    for item in instances.keys():
        class_probabilities[item] = instances[item] / total_instances
        
    return class_probabilities
    
# Smooths some "chunk" (subset) of an image by calculating the average
# of all grayscale values in that chunk. Returns the average calculated.
#
# Parameters:
# chunk - the subset of the image to smooth    
def smooth_chunk(chunk):

    average = 0
    for entry in chunk:
        average += entry
    average = (int)(average / len(chunk))
    
    return average
 
# Builds the vocabulary for each decision class present in the data set and the 
# shared vocabulary for the entire data set. Also calculates the total number of words that
# appear in each decision class. 
#
# Parameters:
# df - the data frame
# vocabulary - the vocabulary dictionary that is shared over all decision classes (i.e. the entire data set)
# class_vocabulary - dictionary object that holds the vocabulary for each decision class present in the data set
# words_per_class - dictionary object that holds counts for how many words appear in each decision class
def build_vocabulary(df, vocabulary, class_vocabulary, words_per_class):
    
    # The index at which actual data begins (grayscale values)
    data_index = 2
    # The index at which the decision class (digit) appears
    category_index = 1
    # The size of a "chunk" (subset) of an image. This is an 8 pixel subset of an image.
    chunk_size = 8
    # 28 x 28 pixel images
    pixels = 28
    
    for row in df.itertuples():
        category = row[category_index]
        
        data = row[data_index:]
        # Calculate the number of chunks that we will iterate over
        num_chunks = (int)(pixels * pixels / 98)
        
        for i in range(0, len(data), num_chunks):
            # Grab a chunk and smooth it
            chunk = data[i: i + chunk_size]
            average = smooth_chunk(chunk)
            
            vocabulary[average] += 1
            class_vocabulary[category][average] += 1
            words_per_class[category] += 1
        
# Given some row in the data set, calculates the probability of that row belonging to some particular decision class.
#
# Parameters: 
# word_list - the row in the data set
# class_vocabulary - the vocabulary for the given decision class
# total_class_words - the number of words that appear in the given decision class
# total_words - the number of words that appear throughout the shared vocabulary
# class_probability - the probability of the decision class appearing in the training set
# alpha - constant value used to avoid zero probabilities    
def find_class_probability(word_list, class_vocabulary, total_class_words, total_words, class_probability, alpha):

    probability = class_probability
    
    # The size of a "chunk" (subset) of an image. This is an 8 pixel subset of an image.
    chunk_size = 8
    # 28 x 28 pixel images
    pixels = 28
    
    # Calculate the number of chunks that we will iterate over
    num_chunks = (int)(pixels * pixels / 98)
    for i in range(0, len(word_list), num_chunks):
        # Grab a chunk and smooth it
        chunk = word_list[i : i + chunk_size]
        average = smooth_chunk(chunk)
        
        if average in class_vocabulary:
            numerator = class_vocabulary[average] + alpha
        else:
            numerator = alpha
            
        denominator = total_class_words + alpha * total_words
        word_probability = numerator / denominator
        
        probability = probability * word_probability
        
        
    return probability
    
# Iterates over some set of test data and makes predictions about which decision class each item belongs to 
# given the class_vocabularies and probabilities calculated.
# 
# Parameters:
# test_df - the test data frame    
# class_vocabularies - the vocabulary for all decision classes
# words_per_class - the number of words that appear in each decision class
# total_words - the number of words that appear throughout the shared vocabulary
# class_probabilities - the probability of each decision class appearing in the training set
# alpha - constant value used to avoid zero probabilities    
def test_class_probability(test_df, class_vocabularies, words_per_class, total_words, class_probabilities, alpha):
    
    prediction_list = []
    data_index = 2
    category_index = 1
    
    for row in test_df.itertuples():
        actual = row[category_index]
        data = row[data_index:]
        
        prob_dict = {}
        prob_list = []
        
        for item in class_vocabularies.keys():
            prob = find_class_probability(data, class_vocabularies[item], words_per_class[item], total_words, class_probabilities[item], alpha)
            prob_dict[item] = prob
            prob_list.append(prob)
            
        predicted_prob = max(prob_list)
        predicted_category = list(prob_dict.keys())[prob_list.index(predicted_prob)]
        pred = (data, actual, predicted_category)
            
        prediction_list.append(pred)
        
    return prediction_list

# Given some list of predictions, calculates the accuracy of the predictions.
#
# Parameters:
# prediction_list - the list of predictions    
def calculate_test_accuracy(prediction_list):

    correct = 0
    total_cases = 0
    
    for prediction in prediction_list:
        (data, actual, predicted) = prediction
        
        if actual == predicted:
            correct += 1
            
        total_cases += 1
        
    accuracy = (correct / total_cases) * 100.0
    return accuracy

# Main function            
def main():

    print("Opening train data frame...")
    train_df = pd.read_csv('data/emnist-digits-train.csv')
    train_df = train_df[:5000]
    
    print("Opening test data frame...")
    test_df = pd.read_csv('data/emnist-digits-test.csv')
    test_df = test_df[:2500]
    
    print("Calculating total number of instances for each decision class...")
    digit_examples = get_number_of_instances(train_df)
    total_examples = total_number_of_instances(digit_examples)
    digit_probabilities = find_decision_class_probabilities(digit_examples, total_examples)
    
    # Declare and define vocabularies
    digit_vocab = {}
    for i in range(10):
        digit_vocab[i] = Counter()
    
    words_per_class = Counter()
    vocab = Counter()
    
    print("Building class vocabularies...")
    build_vocabulary(train_df, vocab, digit_vocab, words_per_class)
    
    total_words = len(vocab.keys())
    
    alpha = 1.0
    prediction_list = []
    
    print("Generating predictions...")
    for digit in digit_vocab.keys():
        prediction_list.extend(test_class_probability(test_df, digit_vocab, words_per_class, total_words, digit_probabilities, alpha))
    
    print("Calculating accuracy...")    
    accuracy = calculate_test_accuracy(prediction_list)
    print('Accuracy: %f%%' % (accuracy))

if __name__ == '__main__':
    main()
    