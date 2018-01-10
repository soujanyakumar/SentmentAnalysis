# -*- coding: utf-8 -*-
import nltk;
from nltk.probability import FreqDist;
# from nltk.probability import DictionaryProbDist;


pos_txt = [('I just want to know the balance of my account.', 'positive'),
              ('My name is kumar', 'positive'),
              ('the number is 123455.', 'positive'),
              ('Can you check the date of the bill?', 'positive'),
              ('Thanks a lot.', 'positive')]
              
              
              
neg_txt = [('Why is there additional charges on my bill in checking account', 'negative'),
              ('what is the transaction fee for', 'negative'),
              ('why is the transaction fee so high', 'negative'),
              ('I dont like the additional charges on my statement', 'negative'),
              ('can I speak to your manager', 'negative')]


CustSvcTxt = []
for (words, sentiment) in pos_txt + neg_txt:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3] 
    CustSvcTxt.append((words_filtered, sentiment))
    
print(CustSvcTxt);
    
    
def get_words_in_CustTxt(CustSvcTxt):
    all_words = []
    for (words, sentiment) in CustSvcTxt:
      all_words.extend(words)
    return all_words
    
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

word_features = get_word_features(get_words_in_CustTxt(CustSvcTxt))
# print(word_features);

document = ['balance', 'my', 'Account'];

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features
    
training_set = nltk.classify.apply_features(extract_features, CustSvcTxt)

classifier = nltk.NaiveBayesClassifier.train(training_set)



# print(classifier);


msg = 'Whats my balance.'
print(nltk.label_probdist.prob('positive'));
print (classifier.classify(extract_features(msg.split())));
# DictionaryProbDist(logprob, normalize=True, log=True)

# print(classifier.show_most_informative_features(32));


# print (classifier.show_most_informative_features(32));

