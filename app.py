import streamlit as st
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import string
punctuation = string.punctuation
def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english')and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)
tfidf= pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
st.title('Email/spam Classifier')
input_sms=st.text_input("Enter the message")
if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")