import pandas as pd
import re
from flask import Flask, request, render_template
from flask import make_response
import spacy
from nltk.corpus import stopwords
from delivery import delivery
from delivery_sentiment import delivery_sentiment
from product import product
from product_sentiment import product_sentiment
from service import service
from service_sentiment import service_sentiment

nlp = spacy.load("en_core_web_sm") 
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.strip()
    text = text.replace('\n', ' ')
    text = re.sub(r'[^\w\s]+|(\d+)', '', text)
    doc = nlp(text)
    text = " ".join([token.lemma_ for token in doc])
    return text

app = Flask(__name__, template_folder='templates')

classifier_delivery, cv_delivery = delivery()
classifier_delivery_sent, cv_delivery_sent = delivery_sentiment()
classifier_product, cv_product = product()
classifier_product_sent, cv_product_sent = product_sentiment()
classifier_service, cv_service = service()
classifier_service_sent, cv_service_sent = service_sentiment()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    df = pd.read_csv(file)
    # df['enreview'] = df['enreview'].apply(clean_text)
    delivery_predictions = classifier_delivery.predict(cv_delivery.transform(df['enreview']))
    delivery_sentiment_predictions = classifier_delivery_sent.predict(cv_delivery_sent.transform(df['enreview']))
    product_predictions = classifier_product.predict(cv_product.transform(df['enreview']))
    product_sentiment_predictions = classifier_product_sent.predict(cv_product_sent.transform(df['enreview']))
    service_predictions = classifier_service.predict(cv_service.transform(df['enreview']))
    service_sentiment_predictions = classifier_service_sent.predict(cv_service_sent.transform(df['enreview']))

    result_df = pd.DataFrame({
        'enreview': df['enreview'],
        'delivery': delivery_predictions,
        'delivery_sentiment': delivery_sentiment_predictions,
        'product': product_predictions,
        'product_sentiment': product_sentiment_predictions,
        'service': service_predictions,
        'service_sentiment': service_sentiment_predictions
    })   

    return render_template('result.html', result=result_df)
@app.route('/download', methods=['POST'])
def download():
    csv_data = request.form['data']
    response = make_response(csv_data)
    response.headers['Content-Disposition'] = 'attachment; filename=result.csv'
    response.headers['Content-type'] = 'text/csv'
    return response
if __name__ == '__main__':
    app.run(port=3000, debug=True)
