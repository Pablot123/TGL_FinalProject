from sklearn.feature_extraction.text import CountVectorizer


import nltk 
from nltk.corpus import stopwords 
import re


stemmer = nltk.stem.SnowballStemmer('english') # Vamos a utlizar el Snowball Stemmer para realizar stemming (nos permite llevar las palabras a una forma estandar). 
nltk.download('stopwords') # Lista de palabras de parada en ingles.

class DataPreprocess:

    def __init__(self, data):
        self.data = data


    def processing_text(self, texto):
        # Paso 1: Remover con un expresión regular carateres especiales (no palabras).
        processed_feature = re.sub(r'\W', ' ', str(texto))
        # Paso 2: Remover ocurrencias de caracteres individuales
        processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
        processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 
        # Paso 3: Remover números (Ocurrencias muy esporádicas en nuestro dataset)
        processed_feature = re.sub(r'[0-9]+', ' ', processed_feature)
        # Paso 4: Simplificar espacios concecutivos a un único espacio entre palabras
        processed_feature = re.sub(' +', ' ', processed_feature)
        # Paso 5: Pasar todo el texto a minúsculas    
        processed_feature = processed_feature.lower()
        # Paso 6: Aplicar stemming. Es una forma de enviar las palabras a una raiz común simplificando de esta manera el vocabulario. 
        #         por ejemplo las palabras (absurdo, absurdos) que estan en el review 2895 seran llevados a la raiz común "absurd"
        #         y de esta forma se evita tener dos palabras diferentes con el mismo significado en nuestro vocabulario.
        processed_feature = " ".join([stemmer.stem(i) for i in processed_feature.split()])
        

        return processed_feature
    
    def convert_data(self):
        raw_data = self.data['review'].values
        texto_procesado = []
        for sentence in raw_data:
            procesado = self.processing_text(sentence)
            texto_procesado.append(procesado)
        return texto_procesado

    def vectroize_data(texto_procesado):
        vectorizer = CountVectorizer(max_features=2500, stop_words=stopwords.words('english'))
        # max_features representa el tamaño del vocabulario. Vamos a permitir 2500 palabras.
        # stop_words le indicamos las palabras de parada para que las ignore en el vocabulario. 

        # Ahora le solicitamos utilizando nuestro conjunto de datos que construya el vocabulario y tambien transforme nuestro texto
        texto_features = vectorizer.fit_transform(texto_procesado).toarray()

        return vectorizer, texto_features

    




