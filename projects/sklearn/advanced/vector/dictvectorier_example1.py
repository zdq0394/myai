from sklearn.feature_extraction import DictVectorizer
measurements = [
    {"City": "Beijing", "Temperature": 36},
    {"City": "Shanghai", "Temperature": 38.0},
    {"City": "Shenzhen", "Temperature": 35.8},
    {"City": "Shenyang", "Temperature": 31.6}
]

vec = DictVectorizer()
a = vec.fit_transform(measurements).toarray()
print(a)

print(vec.get_feature_names())