import pandas as pd
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

data = {
    "text": [
        "NASA confirms water on Mars",
        "Aliens kidnapped my dog",
        "Government releases budget report",
        "Cure for cancer found in bananas"
    ],
    "label": [0, 1, 0, 1]  
}
df = pd.DataFrame(data)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

df['text'] = df['text'].apply(preprocess)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

model = DecisionTreeClassifier()
model.fit(X, y)

pickle.dump(vectorizer, open('vector.pkl', 'wb'))
pickle.dump(model, open('dt.pkl', 'wb'))
print("Files saved: vector.pkl and dt.pkl")
