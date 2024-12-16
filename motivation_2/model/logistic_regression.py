import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# 1. 데이터 로드
# SMSSpamCollection 파일 가정: 첫 컬럼은 라벨('spam' or 'ham'), 두 번째 컬럼은 메시지
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])

# 2. 데이터 전처리
# 라벨: 'spam'/'ham' 그대로 사용
X = df['message'].values
y = df['label'].values

# 3. 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TF-IDF 벡터라이저 학습
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Logistic Regression 모델 학습
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 6. 간단한 성능 평가
train_score = model.score(X_train_vec, y_train)
test_score = model.score(X_test_vec, y_test)
print(f"Train Accuracy: {train_score:.2f}")
print(f"Test Accuracy: {test_score:.2f}")

# 7. 모델 및 벡터라이저 pickle로 저장
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved as model.pkl and vectorizer.pkl.")
