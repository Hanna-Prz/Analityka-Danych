!unzip sms+spam+collection.zip
import pandas as pd
import numpy as np
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def nice_block(msg, true_label, pred_label, idx):
    print("─" * 60)
    print(f"MESSAGE #{idx+1}")
    print(f"LABEL REAL: {'ham' if true_label==1 else 'spam'} | MODEL: {'ham' if pred_label==1 else 'spam'}")
    print("─" * 60)
    print(msg)
    print("─" * 60)
    print()

def remove_punctuation(text):
    import string
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def compute_predict(row):
    msg = row['message']
    clean = row['message_clean']
    if msg == clean:
        return 0.5
    return 0

df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
df['predict'] = 0
df.head()
df['label'] = df['label'].map({'ham': 1, 'spam': -1})
df.head()
df['message'] = df['message'].fillna("")
df['message_clean'] = df['message'].apply(remove_punctuation)
df['predict'] = df.apply(compute_predict, axis=1)
df.head(30)
y = df['label']
X = df['message_clean']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
X = df['message_clean']
y = df['label']
X_train, X_test, y_train, y_test, pred_train, pred_test = train_test_split(
    X, y, df['predict'], test_size=0.2, random_state=42
)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
proba = model.predict_proba(X_test_vec)
adjust_factor = 0.1
proba[:,1] = np.where(pred_test == 0.5, np.minimum(proba[:,1]+adjust_factor, 1), proba[:,1])
proba[:,0] = 1 - proba[:,1]
y_pred_adjusted = np.where(proba[:,1] >= 0.5, 1, -1)
print("Accuracy after adjustment:", accuracy_score(y_test, y_pred_adjusted))
print(classification_report(y_test, y_pred_adjusted))
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "NaiveBayes": MultinomialNB(),
    "SVM": LinearSVC()
}
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    pred = model.predict(X_test_vec)
    print(name, accuracy_score(y_test, pred))
cm = confusion_matrix(y_test, y_pred_adjusted)
sns.heatmap(cm, annot=True, fmt="d")
plt.show()
y_test_s = pd.Series(y_test).reset_index(drop=True)
y_pred_s = pd.Series(y_pred_adjusted).reset_index(drop=True)
X_test_s = pd.Series(X_test).reset_index(drop=True)
df_test = pd.DataFrame({
    'message': X_test_s,
    'true_label': y_test_s,
    'predicted': y_pred_s
})
df_errors = df_test[df_test['true_label'] != df_test['predicted']].reset_index(drop=True)
for i, row in df_errors.iterrows():
    nice_block(row['message'], row['true_label'], row['predicted'], i)



