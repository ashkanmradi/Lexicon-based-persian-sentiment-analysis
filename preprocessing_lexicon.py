from __future__ import unicode_literals
import pandas as pd
from hazm import *
from lexiPersTags import addLexiPers

lexicon = pd.read_excel('377901568617848.xlsx')
lexicon = lexicon.loc[:, 'Persian Translation (Google Translate)':'Negative']
lexicon = lexicon[lexicon['Persian Translation (Google Translate)'].str.contains("[^a-zA-Z]").fillna(False)]
lexicon = lexicon.dropna(axis=0, subset=['Persian Translation (Google Translate)'])
lexicon['sentiment'] = 0
lexiPers = addLexiPers()



for i in range(len(lexicon)):
    if(lexicon.iloc[i,1] == lexicon.iloc[i,2]):
        lexicon.iloc[i, 3] = 0
    elif(lexicon.iloc[i,1] == 1):
        lexicon.iloc[i, 3] = 1
    elif(lexicon.iloc[i,2] == 1):
        lexicon.iloc[i, 3] = -1
    else:
        lexicon.iloc[i, 3] = 4

lexicon = lexicon.dropna(axis=0, subset=['Persian Translation (Google Translate)'])
lexicon = lexicon.sort_values("Persian Translation (Google Translate)")

# lexicon.to_csv(r'lexicon_with_duplicated_values.csv')

duplicates = lexicon[lexicon.duplicated(['Persian Translation (Google Translate)'], keep=False)]
duplicates = duplicates.sort_values("Persian Translation (Google Translate)")

# duplicates.to_csv(r'duplicated_values.csv')


j = 0
handled_duplicates = []
for i in range(len(duplicates)):
    start_index = j
    word = duplicates.iloc[start_index, 0]
    while(word == duplicates.iloc[j, 0]):
        j = j+1
        if(j == len(duplicates)):
            break
    end_index = j
    pos = duplicates.iloc[start_index:end_index, 1].sum()
    neg = duplicates.iloc[start_index:end_index, 2].sum()
    neutral = duplicates.iloc[start_index:end_index, 3].sum()
    if(pos >= neutral and pos > neg):
        case = {'Persian Translation (Google Translate)': word, 'Positive': 1, 'Negative': 0, 'sentiment': 0}
    elif(neg >= neutral and neg > pos):
        case = {'Persian Translation (Google Translate)': word, 'Positive': 0, 'Negative': 1, 'sentiment': 0}
    else:
        case = {'Persian Translation (Google Translate)': word, 'Positive': 0, 'Negative': 0, 'sentiment': 1}
    handled_duplicates.append(case)
    if (j >= len(duplicates)):
        break
handled_duplicates = pd.DataFrame.from_dict(handled_duplicates)

lexicon = lexicon.drop_duplicates(subset="Persian Translation (Google Translate)", keep=False)


for i in range(len(handled_duplicates)):
    if(handled_duplicates.iloc[i,1] == handled_duplicates.iloc[i,2]):
        handled_duplicates.iloc[i, 3] = 0
    elif(handled_duplicates.iloc[i,1] == 1):
        handled_duplicates.iloc[i, 3] = 1
    elif(handled_duplicates.iloc[i,2] == 1):
        handled_duplicates.iloc[i, 3] = -1
    else:
        handled_duplicates.iloc[i, 3] = 4

lexicon = lexicon.append(handled_duplicates, ignore_index=True)
lexicon = lexicon[['Persian Translation (Google Translate)', 'sentiment']]
lexicon = lexicon.append(lexiPers, ignore_index=True)
lexicon = lexicon.drop_duplicates(subset="Persian Translation (Google Translate)", keep='first')
lexicon = lexicon.sort_values("Persian Translation (Google Translate)")
lexicon = lexicon.reset_index(drop=True)

normalizer = Normalizer()
for i in range(len(lexicon)):
    lexicon.iloc[i, 0] = normalizer.normalize(lexicon.iloc[i, 0])

# lexicon.to_csv(r'final_lexicon_without_duplicated_values.csv')

lexicon = lexicon.loc[lexicon['sentiment'] != 0]
lexicon.to_csv(r'final_lexicon_without_duplicated_values_and_zeros.csv')
