# Завантаження бібліотек
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pyngrok import ngrok
from ucimlrepo import fetch_ucirepo

# Закрити попередні тунелі
tunnels = ngrok.get_tunnels()
for tunnel in tunnels:
    ngrok.disconnect(tunnel.public_url)

# Впевнитись, що все ок
print("Усі тунелі закриті")

# Тоді спробувати запустити новий
mlflow_tunnel = ngrok.connect(5000)
mlflow.set_tracking_uri(mlflow_tunnel.public_url)
print("MLflow підключено новим тунелем:", mlflow_tunnel.public_url)

# Завантаження даних
heart = fetch_ucirepo(id=45)
df = pd.concat([heart.data.features, heart.data.targets], axis=1)
df = df.rename(columns={df.columns[-1]: "target"})

# Обробка пропущених значень
imputer = SimpleImputer(strategy="most_frequent")
df[df.columns] = imputer.fit_transform(df)

# Розділення на X та y
X = df.drop("target", axis=1)
y = df["target"]

# Розділення на train і test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Побудова пайплайну
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Параметри для GridSearch
param_grid = {
    "classifier__n_estimators": [50, 100],
    "classifier__max_depth": [5, 10],
    "classifier__min_samples_split": [2, 5]
}

# Авторизація Ngrok
ngrok_token = "30SyHXz4EvR3Ou5hpAkvNTuPK9N_PaKbSJGrg1FcXXqdE3Va"  # Мій токен
ngrok.set_auth_token(ngrok_token)
mlflow_ui_port = 5000
ngrok_tunnel = ngrok.connect(mlflow_ui_port)
mlflow.set_tracking_uri(ngrok_tunnel.public_url)
print("MLflow доступний за адресою:", ngrok_tunnel.public_url)

# Початок експерименту
mlflow.set_experiment("HeartDisease")
with mlflow.start_run():
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy")
    grid_search.fit(X_train, y_train)

    y_pred = grid_search.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", score)
    mlflow.sklearn.log_model(grid_search.best_estimator_, "heart_rf_model")

    print("\nЛіпший score:", round(score, 4))
    print("Найкращі параметри:", grid_search.best_params_)
    print("View run at:", mlflow.active_run().info.run_id)
    print("View experiment at:", ngrok_tunnel.public_url + "/#/experiments/" + str(mlflow.get_experiment_by_name("HeartDisease").experiment_id))

    
