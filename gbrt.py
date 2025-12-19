from utils import load_monk_data, get_encoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

# ---------------------------------------------------------
# 1. CARICAMENTO DATI (Uguale agli altri file)
# ---------------------------------------------------------
# Cambia 'monks-1.train' con monks-2 o monks-3 a seconda del task
train_file = 'monks-1.train'
test_file = 'monks-1.test'

print(f"--- Training GBRT su {train_file} ---")

X_train_raw, y_train = load_monk_data(train_file)
X_test_raw, y_test = load_monk_data(test_file)

# ---------------------------------------------------------
# 2. PREPROCESSING (One-Hot Encoding)
# ---------------------------------------------------------
encoder = get_encoder(X_train_raw)
X_train = encoder.transform(X_train_raw)
X_test = encoder.transform(X_test_raw)

# ---------------------------------------------------------
# 3. CONFIGURAZIONE E TRAINING DEL MODELLO
# ---------------------------------------------------------
# GradientBoostingClassifier implementa il GBRT per la classificazione.
# I parametri chiave da, eventualmente, modificare (tuning) sono:
# - n_estimators: numero di alberi (es. 50, 100, 200)
# - learning_rate: quanto ogni albero corregge gli errori (es. 0.01, 0.1, 0.2)
# - max_depth: profondità massima di ogni albero (per evitare overfitting)

gbrt_model = GradientBoostingClassifier(
    n_estimators=100,     # Numero di alberi
    learning_rate=0.1,    # Tasso di apprendimento standard
    max_depth=3,          # Profondità piccola (funziona bene per ensemble)
    random_state=42       # Per risultati riproducibili
)

gbrt_model.fit(X_train, y_train)

# ---------------------------------------------------------
# 4. VALUTAZIONE
# ---------------------------------------------------------
# Predizione delle classi (0 o 1)
y_pred_train = gbrt_model.predict(X_train)
y_pred_test = gbrt_model.predict(X_test)

# Calcolo Accuracy
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

# Calcolo MSE (utile per confronto con MLP se lo usi in regressione)
test_mse = mean_squared_error(y_test, y_pred_test)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:     {test_acc:.4f}")
print(f"Test MSE:          {test_mse:.4f}")

# (Opzionale) Feature Importance
# Il Boosting è ottimo perché ci dice quali feature contano di più
# print("Feature importances:", gbrt_model.feature_importances_)
