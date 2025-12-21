from gbrt import evaluate_gbrt_classifier, grid_search_gbrt_classifier
from utils import get_encoder, load_monk_data


train_file = "data/MONK/monks-3.train"
test_file = "data/MONK/monks-3.test"

print(f"--- Training GBRT on {train_file} ---")

X_train_raw, y_train = load_monk_data(train_file)
X_test_raw, y_test = load_monk_data(test_file)

encoder = get_encoder(X_train_raw)
X_train = encoder.transform(X_train_raw)
X_test = encoder.transform(X_test_raw)

param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3],
}

search_result = grid_search_gbrt_classifier(param_grid, X_train, y_train, k_folds=5)
best_model = search_result["best_model"]
best_config = search_result["best_config"]

print("Best GBRT config (train CV):", best_config)

metrics = evaluate_gbrt_classifier(best_model, X_train, y_train, X_test, y_test)

print(f"Training Accuracy: {metrics['train_accuracy']:.4f}")
print(f"Test Accuracy:     {metrics['test_accuracy']:.4f}")
print(f"Test MSE:          {metrics['test_mse']:.4f}")
