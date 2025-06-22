import streamlit as st
import pickle
import re
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
ps= PorterStemmer()
stop_words = set(stopwords.words('english'))
vect=TfidfVectorizer()

vector_form=pickle.load(open('vector.pkl','rb'))
load_dt=pickle.load(open('dt.pkl','rb'))

def stemming(content):
   con=re.sub('[^a-zA-Z]',' ',content)
   con=con.lower()
   con=con.split()
   con=[ps.stem(word) for word in con if not word in stop_words]
   con=' '.join(con)
   return con

def fake_news(news):
    news=stemming(news)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    prediction=load_dt.predict(vector_form1)
    return prediction

if __name__ == '__main__':
    st.title("Fake News Classification")
    st.subheader("Input the news content below")

    sentence = st.text_area("Enter your news content here", "some news", height=200)
    predict_btt = st.button("Predict")

    if predict_btt:
        prediction_class = fake_news(sentence)  

        if prediction_class[0] == 0:
            st.success("Reliable")
        elif prediction_class [0]== 1:
            st.warning("Unreliable")
        else:
            st.error("Unexpected prediction result")