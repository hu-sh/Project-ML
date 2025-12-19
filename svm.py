from utils import load_monk_data, get_encoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X_train_raw, y_train = load_monk_data('monks-1.train')
X_test_raw, y_test = load_monk_data('monks-1.test')

encoder = get_encoder(X_train_raw)
X_train = encoder.transform(X_train_raw)
X_test = encoder.transform(X_test_raw)

model = SVC(kernel='rbf')
model.fit(X_train, y_train)
print(f"SVM Accuracy: {accuracy_score(y_test, model.predict(X_test))}")
