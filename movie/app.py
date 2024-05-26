import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import pearsonr
import numpy as np

app = Flask(__name__)

# 데이터 로드
users = pd.read_csv('./users.dat', sep='::', engine='python', names=['사용자ID', '성별', '연령', '직업', '지역'])
ratings = pd.read_csv('./ratings.dat', sep='::', engine='python', names=['사용자ID', '영화ID', '평점', '타임스탬프'])
movies = pd.read_csv('./movies.dat', sep='::', engine='python', names=['영화ID', '영화제목', '장르'])

# 데이터 병합
data = pd.merge(pd.merge(users, ratings), movies)

# 연령대 생성 함수
def generate_ages(y):
    if y < 10:
        return '10대 미만'
    elif y < 20:
        return '10대'
    elif y < 30:
        return '20대'
    elif y < 40:
        return '30대'
    elif y < 50:
        return '40대'
    else:
        return '50대 이상'

data['연령대'] = data['연령'].apply(generate_ages)

# 사용자-아이템 행렬 생성
user_movie_ratings = data.pivot_table(index='사용자ID', columns='영화제목', values='평점')

# 선형 회귀 모델 학습
def train_regression_model():
    X = data[['성별', '연령대', '영화ID']]
    y = data['평점']

    # ColumnTransformer를 사용하여 OneHotEncoding 적용
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['성별', '연령대', '영화ID'])
        ], remainder='passthrough'
    )

    X = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'평균 제곱 오차: {mse}')
    return model, preprocessor

regression_model, preprocessor = train_regression_model()

# 두 영화 간의 상관 계수 계산
def get_movie_correlation(movie1, movie2):
    common_ratings = user_movie_ratings[[movie1, movie2]].dropna()
    if len(common_ratings) > 1:
        correlation, _ = pearsonr(common_ratings[movie1], common_ratings[movie2])
        return correlation
    else:
        return 0  # 공통 평가가 없는 경우 상관 계수 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_movies')
def get_movies():
    movie_list = movies['영화제목'].tolist()
    return jsonify(movie_list)

@app.route('/recommend', methods=['POST'])
def recommend():
    gender = request.form.get('gender')
    age_group = request.form.get('age_group')
    selected_movie = request.form.get('selected_movie')  # 사용자가 선택한 영화 추가

    # 조건에 맞는 데이터 필터링
    filtered_data = data[(data['성별'] == gender) & (data['연령대'] == age_group)]

    # 회귀 모델을 사용하여 평점 예측
    X_filtered = filtered_data[['성별', '연령대', '영화ID']]
    X_filtered = preprocessor.transform(X_filtered)
    filtered_data = filtered_data.copy()  # SettingWithCopyWarning 방지
    filtered_data['예측 평점'] = regression_model.predict(X_filtered)

    # 예측 평점이 높은 순으로 정렬
    top_movies = filtered_data.groupby('영화제목')['예측 평점'].mean().reset_index()
    top_movies = top_movies.sort_values(by='예측 평점', ascending=False).head(5)

    # 유사한 영화 추천 (선택한 영화와 유사도 분석)
    similar_movies = []
    if selected_movie:
        similar_movies = user_movie_ratings.corrwith(user_movie_ratings[selected_movie]).dropna().sort_values(ascending=False).head(6).index.tolist()
        similar_movies = [m for m in similar_movies if m != selected_movie]  # 자신을 목록에서 제거
        similar_movies = similar_movies[:5]  # 상위 5개 유사한 영화 선택

    return jsonify({'top_movies': {'영화제목': top_movies['영화제목'].tolist()}, '유사한 영화': {'영화제목': similar_movies}})

if __name__ == '__main__':
    app.run(debug=True)

