import re
import csv , joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from google.colab import drive
drive.mount('/content/drive')

stop_words_file_path = '/content/drive/MyDrive/data/stopwords.txt'

with open(stop_words_file_path, 'r', encoding='utf-8') as stop_words_file:
    stop_words = stop_words_file.read().splitlines()

    dataset_file_path = '/content/drive/MyDrive/data/DataSet.csv'

preprocessed_data = []
labels = []

with open(dataset_file_path, 'r', encoding='utf-8') as dataset_file:
    csv_reader = csv.reader(dataset_file, delimiter=',', quotechar='"')
    next(csv_reader)  # Skip the header row

    for row in csv_reader:
        if len(row) >= 2:  # Check if the row has the expected number of elements
            text = row[0].strip('"')
            label = row[1].strip('"')

            preprocessed_text = re.sub(r'\W+', ' ', text)  # Remove non-alphanumeric characters
            preprocessed_text = ' '.join([word for word in preprocessed_text.split() if word.lower() not in stop_words])
            preprocessed_data.append(preprocessed_text)
            labels.append(label)


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(preprocessed_data)


classifier = MultinomialNB()
classifier.fit(X, labels)



text_to_classify =" ماهوش عاجبني التخصص "
preprocessed_text_to_classify = re.sub(r'\W+', ' ', text_to_classify)
preprocessed_text_to_classify = ' '.join([word for word in preprocessed_text_to_classify.split() if word.lower() not in stop_words])
vectorized_text_to_classify = vectorizer.transform([preprocessed_text_to_classify])
classification_result = classifier.predict(vectorized_text_to_classify)

if classification_result == 'positive':
    print("The text is classified as positive.")
elif classification_result == 'neutral':
    print("The text is classified as neutral.")
elif classification_result == 'negative':
    print("The text is classified as negative.")
else:
    print("Unknown classification.")



joblib.dump(classifier, 'sentiment-analyzer.pkl')

joblib.dump(vectorizer, "vectorizer.pkl")