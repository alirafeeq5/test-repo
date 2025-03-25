# '''
# SVM not accurate for small dataset
# '''
# import re
# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report

# # Load data (ensure >20 entries with balanced classes)
# df = pd.read_csv("logs.csv")

# # Preprocess logs (clean timestamps, lowercase, etc.)
# def preprocess_log(log):
#     log = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '', log)  # Remove timestamps
#     log = re.sub(r'[^\w\s]', '', log).lower()  # Remove special characters and lowercase
#     return log

# df['processed'] = df['message'].apply(preprocess_log)

# # Vectorize text
# vectorizer = TfidfVectorizer(max_features=1000)
# X = vectorizer.fit_transform(df['processed'])
# y = df['label']

# # Use cross-validation for small datasets
# svm = SVC(kernel='linear', class_weight='balanced', probability=True)  # Enable probability estimates
# scores = cross_val_score(svm, X, y, cv=5)
# print(f"Cross-validated accuracy: {scores.mean():.2f} (±{scores.std():.2f})")

# # Train on the full dataset
# svm.fit(X, y)

# # Test a new log
# # new_log = "<134>1 2023-10-13T14:55:42Z myhostname myapp 1234 - - [exampleSDID@32473 iut=\"3\" eventSource=\"application\" eventID=\"1011\"] User \"jdoe\" logged in successfully from IP 192.168.1.100"
# logs=[
# "<134>1 2023-10-13T14:55:42Z myhostname myapp 1234 - - [exampleSDID@32473 iut=\"3\" eventSource=\"application\" eventID=\"1011\"] User \"jdoe\" logged in successfully from IP 192.168.1.100",
# "CEF:0|Elastic|Vaporware|1.0.0-alpha|18|Authentication|low|message=This event is padded with whitespace dst=192.168.1.2 src=192.168.3.4"  ,
# "2025/01/19 17:34:43 [error] 1633474#1633474: *11485 client intended to send too large body: 10485761 bytes, client: 185.91.69.110, server: backend.bluscout.com, request: POST / HTTP/1.1, host: 16.171.133.191",
# "217.168.17.5 - - [17/May/2015:08:05:34 +0000] \"GET /downloads/product_1 HTTP/1.1\" 200 490 \"-\" \"Debian APT-HTTP/1.3 (0.8.10.3)\"",
# "[Mon Dec 05 19:15:57 2005] [error] mod_jk child workerEnv in error state 6",
# ]
# for new_log in logs:
#     processed_log = preprocess_log(new_log)
#     vectorized_log = vectorizer.transform([processed_log])

#     # Get predicted class and probabilities
#     predicted_class = svm.predict(vectorized_log)[0]
#     probabilities = svm.predict_proba(vectorized_log)[0]

#     print(f"Predicted label: {predicted_class}")
#     print(f"Class probabilities: {dict(zip(svm.classes_, probabilities))}")


'''
Random Forest not accurate for small dataset but most advised on
'''
# import re
# import pandas as pd
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report

# # Load data
# df = pd.read_csv("logs.csv")

# # Preprocess logs
# def preprocess_log(log):
#     log = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '', log)  # Remove timestamps
#     log = re.sub(r'[^\w\s]', '', log).lower()  # Remove special characters and lowercase
#     return log

# df['processed'] = df['message'].apply(preprocess_log)

# # Vectorize text
# vectorizer = TfidfVectorizer(max_features=1000)
# X = vectorizer.fit_transform(df['processed'])
# y = df['label']

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )

# # Train Random Forest with class balancing
# rf = RandomForestClassifier(
#     n_estimators=100,  # Number of trees in the forest
#     class_weight='balanced',  # Handle imbalanced classes
#     random_state=42
# )
# rf.fit(X_train, y_train)

# # Evaluate
# y_pred = rf.predict(X_test)
# # print(classification_report(y_test, y_pred))

# # Cross-validated accuracy
# cv = StratifiedKFold(n_splits=3)  # Use stratified k-fold
# scores = cross_val_score(rf, X, y, cv=cv)
# print(f"Cross-validated accuracy: {scores.mean():.2f} (±{scores.std():.2f})")

# # Test a new log
# # new_log = "<134>1 2023-10-13T14:55:42Z myhostname myapp 1234 - - [exampleSDID@32473 iut=\"3\" eventSource=\"application\" eventID=\"1011\"] User \"jdoe\" logged in successfully from IP 192.168.1.100"

# logs=[
# "<134>1 2023-10-13T14:55:42Z myhostname myapp 1234 - - [exampleSDID@32473 iut=\"3\" eventSource=\"application\" eventID=\"1011\"] User \"jdoe\" logged in successfully from IP 192.168.1.100",
# "CEF:0|Elastic|Vaporware|1.0.0-alpha|18|Authentication|low|message=This event is padded with whitespace dst=192.168.1.2 src=192.168.3.4"  ,
# "2025/01/19 17:34:43 [error] 1633474#1633474: *11485 client intended to send too large body: 10485761 bytes, client: 185.91.69.110, server: backend.bluscout.com, request: POST / HTTP/1.1, host: 16.171.133.191",
# "217.168.17.5 - - [17/May/2015:08:05:34 +0000] \"GET /downloads/product_1 HTTP/1.1\" 200 490 \"-\" \"Debian APT-HTTP/1.3 (0.8.10.3)\"",
# "[Mon Dec 05 19:15:57 2005] [error] mod_jk child workerEnv in error state 6",
# ]
# for new_log in logs:
#     processed_log = preprocess_log(new_log)
#     vectorized_log = vectorizer.transform([processed_log])

#     # Get predicted class and probabilities
#     predicted_class = rf.predict(vectorized_log)[0]
#     probabilities = rf.predict_proba(vectorized_log)[0]

#     print(f"Predicted label: {predicted_class}")
#     print(f"Class probabilities: {dict(zip(rf.classes_, probabilities))}")


'''
Logistic Regression most accurate for small dataset but not advised for huge datasets(logs)
'''
# import re
# import pandas as pd
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report

# # Load data
# df = pd.read_csv("logs.csv")

# # Preprocess logs
# def preprocess_log(log):
#     # log = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '', log)  # Remove timestamps
#     # log = re.sub(r'[^\w\s]', '', log).lower()  # Remove special characters and lowercase
#     return log

# df['processed'] = df['message'].apply(preprocess_log)

# # Vectorize text
# vectorizer = TfidfVectorizer(max_features=1000)
# X = vectorizer.fit_transform(df['processed'])
# y = df['label']

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )

# # Train Logistic Regression with class balancing
# logreg = LogisticRegression(
#     class_weight='balanced',  # Handle imbalanced classes
#     max_iter=1000,  # Increase iterations for convergence
#     random_state=42
# )
# logreg.fit(X_train, y_train)

# # Evaluate
# y_pred = logreg.predict(X_test)
# print(classification_report(y_test, y_pred))

# # Cross-validated accuracy
# cv = StratifiedKFold(n_splits=3)  # Use stratified k-fold
# scores = cross_val_score(logreg, X, y, cv=cv)
# print(f"Cross-validated accuracy: {scores.mean():.2f} (±{scores.std():.2f})")

# # Test a new log
# # new_log = "<134>1 2023-10-13T14:55:42Z myhostname myapp 1234 - - [exampleSDID@32473 iut=\"3\" eventSource=\"application\" eventID=\"1011\"] User \"jdoe\" logged in successfully from IP 192.168.1.100"
# # new_log = "CEF:0|Elastic|Vaporware|1.0.0-alpha|18|Authentication|low|message=This event is padded with whitespace dst=192.168.1.2 src=192.168.3.4"  # Replace with any log message
# # new_log = "2025/01/19 17:34:43 [error] 1633474#1633474: *11485 client intended to send too large body: 10485761 bytes, client: 185.91.69.110, server: backend.bluscout.com, request: POST / HTTP/1.1, host: 16.171.133.191"

# logs=[
# "<134>1 2023-10-13T14:55:42Z myhostname myapp 1234 - - [exampleSDID@32473 iut=\"3\" eventSource=\"application\" eventID=\"1011\"] User \"jdoe\" logged in successfully from IP 192.168.1.100",
# "CEF:0|Elastic|Vaporware|1.0.0-alpha|18|Authentication|low|message=This event is padded with whitespace dst=192.168.1.2 src=192.168.3.4"  ,
# "2025/01/19 17:34:43 [error] 1633474#1633474: *11485 client intended to send too large body: 10485761 bytes, client: 185.91.69.110, server: backend.bluscout.com, request: POST / HTTP/1.1, host: 16.171.133.191",
# "217.168.17.5 - - [17/May/2015:08:05:34 +0000] \"GET /downloads/product_1 HTTP/1.1\" 200 490 \"-\" \"Debian APT-HTTP/1.3 (0.8.10.3)\"",
# "[Mon Dec 05 19:15:57 2005] [error] mod_jk child workerEnv in error state 6",
# ]
# for new_log in logs:
#     processed_log = preprocess_log(new_log)
#     vectorized_log = vectorizer.transform([processed_log])

#     # Get predicted class and probabilities
#     predicted_class = logreg.predict(vectorized_log)[0]
#     probabilities = logreg.predict_proba(vectorized_log)[0]

#     print(f"Predicted label: {predicted_class}")
#     print(f"Class probabilities: {dict(zip(logreg.classes_, probabilities))}")

# # feature_names = vectorizer.get_feature_names_out()
# # for i, class_name in enumerate(logreg.classes_):
# #     print(f"Top 10 features for class '{class_name}':")
# #     coefs = logreg.coef_[i]
# #     top_features = coefs.argsort()[-10:][::-1]
# #     for feature_idx in top_features:
# #         print(f"{feature_names[feature_idx]}: {coefs[feature_idx]:.4f}")


'''
SGDClassifier not accurate for small dataset but good for large datasets
'''
import re
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier  # Changed import
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("logs.csv")

# Preprocess logs
def preprocess_log(log):
    # log = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '', log)  # Remove timestamps
    # log = re.sub(r'[^\w\s]', '', log).lower()
    return log

df['processed'] = df['message'].apply(preprocess_log)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['processed'])
y = df['label']

# Use SGDClassifier instead of SVM
sgd = SGDClassifier(
    loss='log_loss',           # Equivalent to linear SVM
    class_weight='balanced',
    alpha=0.0001,           # Regularization strength (similar to 1/C in SVM)
    max_iter=1000,          # Increase iterations for convergence
    random_state=42
)

# Cross-validation
scores = cross_val_score(sgd, X, y, cv=5)
print(f"Cross-validated accuracy: {scores.mean():.2f} (±{scores.std():.2f})")

# Train on full dataset
sgd.fit(X, y)

# Test a new log
# new_log = "<134>1 2023-10-13T14:55:42Z myhostname myapp 1234 - - [exampleSDID@32473 iut=\"3\" eventSource=\"application\" eventID=\"1011\"] User \"jdoe\" logged in successfully from IP 192.168.1.100"
# new_log = "CEF:0|Elastic|Vaporware|1.0.0-alpha|18|Authentication|low|message=This event is padded with whitespace dst=192.168.1.2 src=192.168.3.4"  
# new_log = "2025/01/19 17:34:43 [error] 1633474#1633474: *11485 client intended to send too large body: 10485761 bytes, client: 185.91.69.110, server: backend.bluscout.com, request: POST / HTTP/1.1, host: 16.171.133.191"
# new_log = "217.168.17.5 - - [17/May/2015:08:05:34 +0000] \"GET /downloads/product_1 HTTP/1.1\" 200 490 \"-\" \"Debian APT-HTTP/1.3 (0.8.10.3)\""
# new_log = "[Mon Dec 05 19:15:57 2005] [error] mod_jk child workerEnv in error state 6"

# logs=[
# "<134>1 2023-10-13T14:55:42Z myhostname myapp 1234 - - [exampleSDID@32473 iut=\"3\" eventSource=\"application\" eventID=\"1011\"] User \"jdoe\" logged in successfully from IP 192.168.1.100",
# "CEF:0|Elastic|Vaporware|1.0.0-alpha|18|Authentication|low|message=This event is padded with whitespace dst=192.168.1.2 src=192.168.3.4"  ,
# "2025/01/19 17:34:43 [error] 1633474#1633474: *11485 client intended to send too large body: 10485761 bytes, client: 185.91.69.110, server: backend.bluscout.com, request: POST / HTTP/1.1, host: 16.171.133.191",
# "217.168.17.5 - - [17/May/2015:08:05:34 +0000] \"GET /downloads/product_1 HTTP/1.1\" 200 490 \"-\" \"Debian APT-HTTP/1.3 (0.8.10.3)\"",
# "[Mon Dec 05 19:15:57 2005] [error] mod_jk child workerEnv in error state 6",
# ]
logs=[
"127.0.0.1 - - [16/Feb/2025:14:30:12] \"GET /index.html HTTP/1.1\" 200 512 \"-\" \"Mozilla/5.0\"",
"192.168.1.50 - - [16/Feb/2025:14:31:05] \"POST /login HTTP/1.1\" 403 256 \"-\" \"Chrome/110.0\"",
"10.0.0.5 - - [16/Feb/2025:14:32:21] \"GET /styles.css HTTP/1.1\" 304 0 \"-\" \"Edge/115.0\"",
"203.0.113.90 - - [16/Feb/2025:14:33:18] \"GET /admin HTTP/1.1\" 401 128 \"-\" \"Python-urllib/3.9\"",
"172.16.5.20 - - [16/Feb/2025:14:34:33] \"GET /api/status HTTP/1.1\" 500 512 \"-\" \"PostmanRuntime/7.26\"",
"8.8.4.4 - - [16/Feb/2025:14:35:00] \"GET /favicon.ico HTTP/1.1\" 404 128 \"-\" \"Googlebot\"",
"192.168.10.1 - - [16/Feb/2025:14:36:45] \"POST /upload HTTP/1.1\" 413 1024 \"-\" \"Java-http-client/11\"",
"10.10.10.20 - - [16/Feb/2025:14:37:12] \"GET /metrics HTTP/1.1\" 200 4096 \"-\" \"Prometheus/2.34\"",
]
for new_log in logs:
    processed_log = preprocess_log(new_log)
    vectorized_log = vectorizer.transform([processed_log])

    # Get prediction
    predicted_class = sgd.predict(vectorized_log)[0]
    print(f"Predicted label: {predicted_class}")

    # For probability estimates (requires different loss function)
    probabilities = sgd.predict_proba(vectorized_log)[0]
    print(f"Class probabilities: {dict(zip(sgd.classes_, probabilities))}")
