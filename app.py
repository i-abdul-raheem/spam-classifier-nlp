from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd
import re
import joblib

class SpamDetection:
    def __init__(self, dataset="./sms+spam+collection/SMSSpamCollection", max_features=2500):
        self.cv = CountVectorizer(max_features=max_features)
        self.ps = PorterStemmer()
        self.dataset = pd.read_csv(dataset, sep="\t", names=["labels", "messages"])
        self.corpus = []
        self.X = None
        self.y = pd.get_dummies(self.dataset["labels"]).iloc[:,1].values
        self.model = None
        
    def process_text(self):
        print("[!] Processing the dataset...")
        for i in range(len(self.dataset["messages"])):
            msg = self.dataset["messages"][i].lower()
            msg = re.sub("[^a-z]", " ", msg).split()
            msg = ' '.join([self.ps.stem(word) for word in msg if word not in set(stopwords.words("english"))])
            self.corpus.append(msg)
        print("[+] Dataset processed...")
            
    def get_bow(self):
        print("[!] Extracting bag of words...")
        self.X = self.cv.fit_transform(self.corpus).toarray()
        print("[+] BoW extracted")
        
    def train(self):
        self.process_text()
        self.get_bow()
        X_train, self.X_test, y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.8, random_state=45)
        print("[!] Starting training...")
        self.model = MultinomialNB().fit(X_train, y_train)
        print("[+] Trainging completed")
        print("[!] Saving the model...")
        joblib.dump(self.model, "SpamDetectionModel.pkl")
        joblib.dump(self.cv, "BoW.pkl")
        print("[+] Model saved")
        self.test()
    
    def test(self):
        y_pred = self.model.predict(self.X_test)
        print("Accuracy: " + str(accuracy_score(self.y_test, y_pred)))
    
    def predict(self, x):
        txt = " ".join([self.ps.stem(word) for word in re.sub("[^a-z]", " ", x.lower()).split() if word not in stopwords.words("english")])
        if not self.model:
            try:
                self.model = joblib.load("SpamDetectionModel.pkl")
            except FileNotFoundError:
                print("[!] Model not found. Training model...")
                self.train()
        try:
            cv = self.cv.transform([txt]).toarray()
        except:
            try:
                self.cv = joblib.load("BoW.pkl")
                cv = self.cv.transform([txt]).toarray()
            except FileNotFoundError:
                print("[!] Vectorizer not found. Re-training model...")
                self.train()
                cv = self.cv.transform([txt]).toarray()
            
        prediction = self.model.predict(cv)[0]
        label = "Spam" if prediction == 1 else "Ham"
        confidence = self.model.predict_proba(cv)[0][prediction]
        return label, str(round(float(max(confidence[0])), 2) * 100) + "%"
        
if __name__ == "__main__":
    sd = SpamDetection()
    result = sd.predict("Congratulations! You've won a $1000 Walmart gift card. Click here!")
    print(result[0] + ": " + result[1])