import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


# Open and read the text file
with open('the-call-of-cthulhu-h-p-lovecraft.txt', 'r', encoding='utf-8') as file:
    text = file.read()

import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Convert to lowercase
text = text.lower()

# Remove punctuation
text = ''.join([char for char in text if char not in string.punctuation])

# Tokenization
tokens = word_tokenize(text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens if word not in stop_words]

# Stemming (optional)
stemmer = PorterStemmer()
tokens = [stemmer.stem(word) for word in tokens]

# Join the tokens back into a single string
processed_text = ' '.join(tokens)

sid = SentimentIntensityAnalyzer()
sentiment_scores = sid.polarity_scores(text)

print(sentiment_scores)