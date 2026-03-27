import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score ,confusion_matrix
import dagshub

dagshub.init(repo_owner="rahulpatel16092005", repo_name="mlflow-dagshub", mlflow=True) 
mlflow.set_tracking_uri("https://dagshub.com/rahulpatel16092005/mlflow-dagshub.mlflow") 


load_iris = load_iris()
X = load_iris.data
y = load_iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_depth = 5

mlflow.set_experiment("iris-dt")

with mlflow.start_run():
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    mlflow.log_param("max_depth", max_depth)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    cm=confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = range(len(load_iris.target_names))
    plt.xticks(tick_marks, load_iris.target_names, rotation=45)
    plt.yticks(tick_marks, load_iris.target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)

    mlflow.sklearn.log_model(model, "random_forest_model")