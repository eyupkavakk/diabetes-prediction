from sklearn.model_selection import  GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import joblib
from utils import load_and_inspect_data, preprocess_data

file_path = "diabetes.csv"  
df = load_and_inspect_data(file_path)

if df is not None:  

    # Veri Ön İşleme
    outlier_features = ["Age", "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction"]
    X_train, X_test, y_train, y_test = preprocess_data(df, "Outcome", outlier_features=outlier_features, test_size=0.2, random_state=42)

    # Veri Standardizasyonu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelleri Eğitme ve Değerlendirme
    models = {
        "Logistic Regression": (LogisticRegression(),
                               {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}),
        "SVM": (SVC(probability=True),
                {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}),
        "Decision Tree": (DecisionTreeClassifier(),
                          {'max_depth': [3, 5, 7, None], 'criterion': ['gini', 'entropy']}),
        "Random Forest": (RandomForestClassifier(),
                          {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, None], 'criterion': ['gini', 'entropy']})
    }

    best_model = None
    best_accuracy = 0

    print("\nModel Performansları:")
    for name, (model, param_grid) in models.items():
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        print(f"\n{name}:")
        print("En iyi parametreler:", grid_search.best_params_)
        print("En iyi doğruluk skoru:", grid_search.best_score_)

        y_pred = grid_search.best_estimator_.predict(X_test_scaled)
        print("Test Doğruluğu:", accuracy_score(y_test, y_pred))
        print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
        print("Karmaşıklık Matrisi:\n", confusion_matrix(y_test, y_pred))

        # ROC Eğrisi
        y_pred_proba = grid_search.best_estimator_.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC eğrisi (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{name} ROC Eğrisi')
        plt.legend(loc="lower right")
        plt.show()

        if accuracy_score(y_test, y_pred) > best_accuracy:
            best_accuracy = accuracy_score(y_test, y_pred)
            best_model = grid_search.best_estimator_

    # En iyi modeli kaydet
    if best_model:
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(best_model, "early_diabetes_model.pkl")
        print("\nEn iyi model ve ölçekleyici kaydedildi.")