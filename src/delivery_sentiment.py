import pandas as pd

from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def delivery_sentiment():
    dfmodel = pd.read_csv('E:/STUDENTS-UNIVERSITY/FRESHER_TERM_2/CAPSTONE PJ 2/data/df_sample1.csv')
    X = dfmodel['enreview']
    # y = dfmodel['delivery']
    y2 = dfmodel['delivery_sent']
    X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size = 0.3,random_state=188)

    cv_delivery_sent = CountVectorizer()
    X_train = cv_delivery_sent.fit_transform(X_train)
    X_test = cv_delivery_sent.transform(X_test)

    classifier_delivery_sent = SVC(kernel = 'linear', random_state = 0)
    classifier_delivery_sent.fit(X_train, y_train)
    return classifier_delivery_sent,cv_delivery_sent