import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import gradio as gr

df = pd.read_csv("train.csv")
categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df['is_toxic'] = df[categories].sum(axis=1) > 0
df['is_toxic'] = df['is_toxic'].astype(int)
df = df[['comment_text', 'is_toxic']]

X_train, X_test, y_train, y_test = train_test_split(df['comment_text'], df['is_toxic'], test_size=0.2)

vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

print(classification_report(y_test, y_pred))

def predict(comment):
    vec = vectorizer.transform([comment])
    pred = model.predict(vec)[0]
    return "Toxic 🚨" if pred else "Not Toxic ✅"

gr.Interface(predict, gr.Textbox(), "text", title="TF-IDF Toxic Comment Detector").launch()
