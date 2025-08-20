# 📩 SMS Spam Detection using Multinomial Naive Bayes

This project is a Python-based SMS spam classifier that uses NLP techniques and a Multinomial Naive Bayes model to detect spam messages with high accuracy. It is trained on the [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

---

## 🚀 Features

- Text preprocessing: lowercase, regex cleanup, stop word removal, stemming
- Bag of Words (BoW) feature extraction
- Spam/Ham classification using Multinomial Naive Bayes
- Confidence score for predictions
- Auto fallback to train if model or vectorizer is missing

---

## 📁 Dataset

- **Source**: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Format**: Tab-separated file with two columns: `label` (ham/spam) and `message`.

---

## 🛠️ Requirements

- Python 3.6+
- Install dependencies:

```bash
pip install -r requirements.txt
```

Also, make sure to download NLTK stopwords:
```python
import nltk
nltk.download('stopwords')
```

---

## 🧠 How it works

1. **Training**:
   - Preprocess the text (clean, tokenize, stem, remove stopwords)
   - Convert to numerical features using `CountVectorizer`
   - Train a `MultinomialNB` classifier
   - Save the trained model and vectorizer to disk

2. **Prediction**:
   - Preprocess the input
   - Vectorize using saved BoW model
   - Predict class and return confidence score

---

## 📦 Usage

```python
from spam_detector import SpamDetection

sd = SpamDetection()
label, confidence = sd.predict("Congratulations! You've won a $1000 Walmart gift card. Click here!")
print(f"{label}: {confidence}")
```

---

## 🧪 Example Output

```
Spam: 97.0%
Ham: 93.5%
```

---

## 📂 File Structure

```
.
├── sms+spam+collection/
│   └── SMSSpamCollection
├── app.py
├── SpamDetectionModel.pkl
├── BoW.pkl
└── README.md
```

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙌 Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- NLTK for natural language processing tools
- scikit-learn for model training and evaluation

---
