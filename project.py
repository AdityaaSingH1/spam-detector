from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import streamlit as st

readdata = pd.read_csv(
    'spam.csv', encoding='latin-1')
print(readdata)
df = readdata[['v1', 'v2']]
df.columns = ['label', 'text']
print(df.head())
df['label'] = df['label'].map({'ham': 0, 'spam': 1})


X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df["label"], random_state=69, test_size=0.2
)


@st.cache_resource
def train_model():
    tfidf = TfidfVectorizer()
    Xtraindata = tfidf.fit_transform(X_train)
    Xtestdata = tfidf.transform(X_test)

    model = MultinomialNB()
    model.fit(Xtraindata, y_train)

    ypred = model.predict(Xtestdata)
    acc = accuracy_score(y_test, ypred)

    return model, ypred, acc, tfidf


model, ypred, acc, tfidf = train_model()

st.set_page_config(page_title="spam dectector website ",
                   page_icon="📩",
                   layout="centered")
st.title("Email Spam Detector ")
st.write("Enter a message below to check whether it is Spam or Not")
user_input = st.text_area(" Enter your message:")
if st.button("Check"):
    if user_input.strip() == "":
        st.error("please enter any text to be checked")
    else:
        test = [user_input.lower()]
        test_data = tfidf.transform(test)
        result = (model.predict(test_data))
        prob = model.predict_proba(test_data)
        if result[0] == 0:
            st.success("not spam")
            st.write("Confidence:")
            st.write(
                f" Not Spam and the percent of the output is: {prob[0][0]*100:.2f}")
        else:
            st.error("spam")
            st.write("Confidence:")
            st.write(
                f"Spam and the percent of the output is: {prob[0][1]*100:.2f}")
