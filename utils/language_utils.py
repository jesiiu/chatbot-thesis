import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
import spacy
from unidecode import unidecode
import re

#Klasa zawierająca funkcje pomocnicze w przetważaniu NLP
#Można ją rozbudowywać o kolejne funkcjonalności
class Utils:
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.nlp = spacy.load('pl_core_news_md')

    def tokenize_word(self, input_sentence):
        return nltk.word_tokenize(input_sentence)

    def stem_word(self, word):
        return self.stemmer.stem(word.lower())

    def bag_of_words(self, tokenized_sentence, words):
        sentence_words = [self.stem_word(word) for word in tokenized_sentence]
        bag = np.zeros(len(words), dtype=np.float32)
        for i, w in enumerate(words):
            if w in sentence_words:
                bag[i] = 1
        return bag

    def get_city_name(self, sentence):
        doc = self.nlp(sentence)
        if len(doc.ents) < 1:
            return None, None
        for ent in doc.ents:
            if ent.label_ == 'geogName' or ent.label_ == 'placeName':
                entry = ent.lemma_
                city = ent.lemma_
                city = self.__remove_polish_chars(city)
                city = self.__get_english_name(city)
                return city, entry

    def __remove_polish_chars(self, word):
        return unidecode(word)

    def __get_english_name(self, word):
        if (str(word).endswith('yt')):
            return word[:-2] + 'id'
        if (str(word).endswith('yn')):
            return word[:-2] + 'on'
        return word

    def get_order_id(self, sentence):
        match = re.search(r'(\d{1,15})', sentence)
        if match:
            return match.group(0).strip().lstrip().rstrip()
        else:
            return 0
