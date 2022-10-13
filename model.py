from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class MLmodel:

    def __init__(self) -> None:
        pass

    def create_model():

        model = MultinomialNB()

        return model

    def train(model, X_train, y_train):

        model.fit(X_train, y_train)

        return model
