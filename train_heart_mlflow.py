# Імпорти
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from ucimlrepo import fetch_ucirepo
import time
import mlflow
import mlflow.sklearn
from pyngrok import ngrok

# Дані: завантаження з UCI
heart = fetch_ucirepo(id=45)
df = pd.concat([heart.data.features, heart.data.targets], axis=1)

# Ціль
df.rename(columns={"num": "target"}, inplace=True)
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

print("Розподіл цільової змінної:\n", df["target"].value_counts())

# 🧹 Перевірка пропущених значень
print("Пропущені значення:\n", df.isna().sum())

# Видаляємо пропущені або заповнюємо
df.dropna(inplace=True)  # або df.fillna(...)

# Категорії вручну
categorical_features = ['cp', 'restecg', 'slope', 'thal', 'sex', 'fbs', 'exang']
for col in categorical_features:
    df[col] = df[col].astype("category")

# Числові ознаки
numeric_features = df.drop(columns=["target"] + categorical_features).columns.tolist()

# Ознаки та ціль
X = df.drop(columns="target")
y = df["target"]

# Розбиття
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Препроцесор
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

# Параметри
params = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [5, None],
    "classifier__min_samples_split": [2, 5],
}

# Пошук
grid_search = GridSearchCV(pipeline, param_grid=params, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

# Оцінка
y_pred = grid_search.best_estimator_.predict(X_test)
print("\nКласифікаційний звіт:\n", classification_report(y_test, y_pred))
print(f"Ліпший score: {grid_search.best_score_:.4f}")
print(f"Найкращі параметри: {grid_search.best_params_}")

# MLflow
mlflow_ui_port = 5000
ngrok.kill()
time.sleep(2)
ngrok_tunnel = ngrok.connect(mlflow_ui_port)

ngrok.set_auth_token("ТУТ_ТВІЙ_NGROK_TOKEN")
ngrok_tunnel = ngrok.connect(mlflow_ui_port)
mlflow.set_tracking_uri(ngrok_tunnel.public_url)
print("MLflow UI доступний за адресою:", ngrok_tunnel.public_url)

mlflow.set_experiment("HeartDisease")
with mlflow.start_run():
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_score", grid_search.best_score_)
    mlflow.sklearn.log_model(grid_search.best_estimator_, "model")
