from sklearn.feature_extraction.text import  CountVectorizer
import nltk

sent1 = "The cat is walking in the bedrood."
sent2 = "A dog was running across the kitchen."
count_vec = CountVectorizer()
sents = [sent1, sent2]

sents_array = count_vec.fit_transform(sents).toarray()
print(count_vec.get_feature_names())
print(sents_array)

tokens_1 = nltk.word_tokenize(sent1)
print(tokens_1)
tokens_2 = nltk.word_tokenize(sent2)
print(tokens_2)

vocab_1 = sorted(set(tokens_1))
print(vocab_1)
vocab_2 = sorted(set(tokens_2))
print(vocab_2)

stemmer = nltk.stem.PorterStemmer()
stem_1 = [stemmer.stem(t) for t in tokens_1]
print(stem_1)
stem_2 = [stemmer.stem(t) for t in tokens_2]
print(stem_2)

