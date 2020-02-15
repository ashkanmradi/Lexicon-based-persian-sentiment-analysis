from __future__ import unicode_literals
from hazm import *
import pandas as pd
import numpy as np
from PersianStemmer import PersianStemmer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from plot_confusion_matrix import plot_confusion_matrix



shifters = pd.read_csv('/home/ashkan/Documents/PycharmProjects/sentiment_analysis/shifters.csv')
intensifiers = pd.read_csv('/home/ashkan/Documents/PycharmProjects/sentiment_analysis/intensifiers.csv')
lexicon = pd.read_csv('/home/ashkan/Documents/PycharmProjects/sentiment_analysis/final_lexicon_without_duplicated_values_and_zeros.csv')
lexicon = lexicon.sort_values("Persian Translation (Google Translate)")
lexicon = lexicon.reset_index(drop=True)


normalizer = Normalizer()
stemmer = PersianStemmer()


for i in range(len(shifters)):
    shifters.iloc[i, 0] = normalizer.normalize(shifters.iloc[i, 0])

for i in range(len(intensifiers)):
    intensifiers.iloc[i, 0] = normalizer.normalize(intensifiers.iloc[i, 0])


'''
f = open(r'/home/ashkan/Documents/PycharmProjects/sentiment_analysis/wor2vec_cbow300d.txt', 'r')
w2v_cbow = f.readlines()
f.close()

w2v_cbow = w2v_cbow[1:]

vector_words = []
for i in range(len(w2v_cbow)):
    temp = w2v_cbow[i].split()
    vector_words.append(temp[0])

lexicon_words = lexicon['Persian Translation (Google Translate)'].to_list()

intersect = list(set(lexicon_words) & set(vector_words))

lst = []
sentiment = []
for i in range(len(lexicon)):
    if(lexicon.iloc[i, 1] in intersect):
        lst.append(lexicon.iloc[i, 1])
        sentiment.append(lexicon.iloc[i,2])

lexicon = pd.DataFrame(list(zip(lst, sentiment)), columns =['Persian Translation (Google Translate)', 'sentiment'])

w2v_cbow_dict = {}
normalizer = Normalizer()
for i in range(len(w2v_cbow)):
    temp = w2v_cbow[i]
    temp = temp.split()
    temp2 = np.array(temp[1:], dtype=float)
    temp[0] = normalizer.normalize(temp[0])
    w2v_cbow_dict[temp[0]] = temp2
'''


data = pd.read_excel('/home/ashkan/Documents/PycharmProjects/sentiment_analysis/soha.xlsx')

sentences = data['sentences'].tolist()
label = data['Label_1'].tolist()
pred = []



cntr = 0
for sentence in sentences:
    sentence = normalizer.normalize(sentence)
    print(sentence)
    sentence = "".join(c for c in sentence if c not in ('!', '.', ':', '،', '؛', '/', '\''))

    ##################################
    # words = sentence.split()
    words = word_tokenize(sentence)
    #################################

    score = 0
    last_word = ""
    last_sentiment = 0
    for word in words:
        stemmed_word = stemmer.run(word)
        # print(word)
        word_sentiment = 0
        # if lexicon['Persian Translation (Google Translate)'].str.contains(word).any():
        if (lexicon['Persian Translation (Google Translate)'] == stemmed_word).any():
            word_index = lexicon.index[lexicon['Persian Translation (Google Translate)'] == word]
            word_sentiment = lexicon.iloc[word_index, 1].to_list()
            # print('****   ', stemmed_word, ':  ', word_sentiment, '     *****')
        # if shifters['shifters'].str.contains(word).any():
        if (shifters['shifters'] == word).any():
            if(last_sentiment > 0):
                score = score - last_sentiment - 1
            elif(last_sentiment < 0):
                score = score - last_sentiment + 1
            else:
                pass

        # this part is fucking slow (else'e if avalie) ==> find nearest word in the lexicon, if the current word is not present in the lexicon #
        # elif(word in w2v_cbow_dict):
        #     wordVec = w2v_cbow_dict[word]
        #     max_similarity = -1
        #     for j in range(len(lexicon)):
        #         temp = lexicon.iloc[j, 0]
        #         dot = np.dot(wordVec, w2v_cbow_dict[temp])
        #         norma = np.linalg.norm(wordVec)
        #         normb = np.linalg.norm(w2v_cbow_dict[temp])
        #         cos = dot / (norma * normb)
        #         if(cos > max_similarity):
        #             max_similarity = cos
        #             most_similar_word = temp
        #     if(max_similarity > 0.7):
        #         word_index = lexicon.index[lexicon['Persian Translation (Google Translate)'] == most_similar_word]
        #         word_sentiment = lexicon.iloc[word_index, 1].to_list()


        if(last_word):
            # if intensifiers['Intensifiers'].str.contains(last_word).any():
            if (intensifiers['Intensifiers'] == last_word).any():
                if(word_sentiment):
                    word_sentiment[0] = word_sentiment[0] * 2

        if(word_sentiment):
            score = score + word_sentiment[0]
            print(word, word_sentiment[0])
        last_word = word
        if(word_sentiment):
            last_sentiment = word_sentiment[0]
        else:
            last_sentiment = 0

    if(score > 0):
        print('real sentiment: ', label[cntr])
        print("\nSentence Score is: ", score, ", The comment is Positive", "\n---------------------------")
        pred.append(1)
    elif(score < 0):
        print('real sentiment: ', label[cntr])
        print("\nSentence Score is: ", score, ", The comment is Negative", "\n---------------------------")
        pred.append(-1)
    else:
        print('real sentiment: ', label[cntr])
        print("\nSentence Score is: ", score, ", The comment is Neutral", "\n---------------------------")
        pred.append(0)

    cntr = cntr + 1

cnf_matrix = confusion_matrix(label, pred, labels=[1, 0, -1])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Positive', 'Neutral', 'Negative'],
                      title='Confusion matrix, without normalization')

plt.show()

# dot = np.dot(w2v_cbow_dict['می‌باشد'], w2v_cbow_dict['می‌باشد'])
# norma = np.linalg.norm(w2v_cbow_dict['می‌باشد'])
# normb = np.linalg.norm(w2v_cbow_dict['می‌باشد'])
# cos = dot / (norma * normb)
# print(cos)
