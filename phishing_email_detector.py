import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def extract_url_count(text):
    if not isinstance(text, str):
        return 0
    urls = re.findall(r'https?://\S+|www\.\S+', text)
    return len(urls)


def main():
    try:
        data = pd.read_csv("phishing_emails.csv")
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return

    if 'email' not in data.columns or 'label' not in data.columns:
        print("Dataset must contain 'email' and 'label' columns.")
        return

    data['email'] = data['email'].fillna('')
    data['url_count'] = data['email'].apply(extract_url_count)

    # TF-IDF Features
    vectorizer = TfidfVectorizer(stop_words='english')
    X_text = vectorizer.fit_transform(data['email'])

    # Add URL count as an extra feature
    url_feat = csr_matrix(data['url_count'].values.reshape(-1, 1))
    X = hstack([X_text, url_feat])

    # Map labels to integers
    label_map = {'safe': 0, 'phishing': 1}
    if data['label'].dtype == object:
        y = data['label'].map(label_map)
    else:
        y = data['label']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model (increase max_iter to ensure convergence)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy*100:.2f}%")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['safe', 'phishing']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Safe', 'Phishing'], yticklabels=['Safe', 'Phishing'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Real-Time Testing
    inv_map = {v: k for k, v in label_map.items()}
    while True:
        email = input("\nEnter Email Text (or type exit): ")
        if email.strip().lower() == "exit":
            break
        transformed = vectorizer.transform([email])
        transformed = hstack([transformed, csr_matrix([[extract_url_count(email)]])])
        prediction = model.predict(transformed)
        label_str = inv_map.get(int(prediction[0]), str(prediction[0]))
        print("\nResult:", label_str.upper())


if __name__ == "__main__":
    main()