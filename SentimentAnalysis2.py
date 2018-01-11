# -*- coding: utf-8 -*-
import nltk;
from nltk.probability import FreqDist;
import numpy as np;
from numpy import loadtxt
import matplotlib.pyplot as plt


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
              ('can I speak to your manager', 'negative'),
              ('Not ok with your service', 'negative')]

# import pandas as pd
# mlb = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/intro_to_python/baseball.csv")                            
# height = mlb['Height'].tolist()
# print(height);



CustSvcTxt = []
for (words, sentiment) in pos_txt + neg_txt:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3] 
    CustSvcTxt.append((words_filtered, sentiment))
    
# print(CustSvcTxt);
    
    
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


# print(get_words_in_CustTxt(CustSvcTxt));
# print(word_features);

document = ['balance', 'my', 'Account', 'number','manager','fees'];

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features
    
training_set = nltk.classify.apply_features(extract_features, CustSvcTxt)

# print(training_set);

classifier = nltk.NaiveBayesClassifier.train(training_set)


# print(classifier);

# "C:\Study\DataScience\CustomerServiceCoversation\Data.csv"

    
messages = [0,""];
msg = 'not happy with your service.'
messages = np.array([0,msg]);
print (classifier.classify(extract_features(msg.split())));


msg = 'whats my balance.'
messages = np.append(1,msg);
print (classifier.classify(extract_features(msg.split())));
print (messages)

msg = 'What is the fees on my account.'
print (classifier.classify(extract_features(msg.split())));


msg = 'can I speak to your manager.'
print (classifier.classify(extract_features(msg.split())));


msg = 'I do not like your service.'
print (classifier.classify(extract_features(msg.split())));


res = classifier.classify(extract_features(msg.split()));

if res == "positive":
    sent = 1
else:
    sent = 0

print(sent);

y = np.array([1.0,1.0,0.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,0.0,1.0])
# x = np.arange(0.0, 1.0, 0.01)
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
# x = 1 + np.sin(2 * np.pi * y)

fig, ax = plt.subplots()
ax.plot(x, y)

ax.set(xlabel='Time', ylabel='Sentiment',
       title='customer Sentiment')
ax.grid()

fig.savefig("test.png")
plt.show()



# DictionaryProbDist(logprob, normalize=True, log=True)

# print(classifier.show_most_informative_features(32));


# print (classifier.show_most_informative_features(32));

