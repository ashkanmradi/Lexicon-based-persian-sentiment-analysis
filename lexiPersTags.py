def addLexiPers():
    import xml.etree.ElementTree as ET
    import pandas as pd

    root = ET.parse('/home/ashkan/Documents/PycharmProjects/sentiment_analysis/LexiPersV1.0/Data/adj-final.xml').getroot()

    sense = []
    label = []
    for synset in root.findall('Synset'):
        sense.append(synset.get('Sense'))
        label.append(int(synset.get('Label')))

    lexiDict = dict(zip(sense, label))

    lexiPers = pd.DataFrame.from_dict(lexiDict, orient='index').T

    key = []
    values = []
    for i in range(lexiPers.shape[1]):
        temp = lexiPers.columns[i]
        x = temp.split(',')
        for j in range(len(x)):
            key.append(x[j])
            values.append(lexiPers.iloc[0, i])

    df = pd.DataFrame(list(zip(key, values)), columns=['Persian Translation (Google Translate)', 'sentiment'])

    return df
