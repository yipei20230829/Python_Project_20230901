import streamlit as st
import joblib

# 載入模型與標準化轉換模型
clf = joblib.load('breast_cance_model.joblib')
scaler = joblib.load('breast_cancer_scaler.joblib')

st.title('依檢驗值預測受檢者是否罹患乳癌（Breast Cancer）')
mean_radius = st.slider('radius (mean):', min_value=6.981, max_value=28.11, value=14.13)
mean_texture = st.slider('texture (mean):', min_value=9.71, max_value=39.28, value=19.29)
mean_perimeter = st.slider('perimeter (mean):', min_value=43.79, max_value=188.5, value=91.97)
mean_area = st.slider('area (mean):', min_value=143.5, max_value=2501.0, value=654.89)
mean_smoothness = st.slider('smoothness (mean):', min_value=0.053, max_value=0.163, value=0.10)
mean_compactness = st.slider('compactness (mean):', min_value=0.019, max_value=0.345, value=0.10)
mean_concavity = st.slider('concavity (mean):', min_value=0.0, max_value=0.427, value=0.09)
mean_concave_points = st.slider('concave points (mean):', min_value=0.0, max_value=0.201, value=0.05)
mean_symmetry = st.slider('symmetry (mean):', min_value=0.106, max_value=0.304, value=0.18)
mean_fractal_dimension = st.slider('fractal dimension (mean):', min_value=0.05, max_value=0.097, value=0.06)
radius_error = st.slider('radius (standard error):', min_value=0.112, max_value=2.873, value=0.41)
texture_error = st.slider('texture (standard error):', min_value=0.36, max_value=4.885, value=1.22)
perimeter_error = st.slider('perimeter (standard error):', min_value=0.757, max_value=21.98, value=2.87)
area_error = st.slider('area (standard error):', min_value=6.802, max_value=542.2, value=40.34)
smoothness_error = st.slider('smoothness (standard error):', min_value=0.002, max_value=0.031, value=0.01)
compactness_error = st.slider('compactness (standard error):', min_value=0.002, max_value=0.135, value=0.03)
concavity_error = st.slider('concavity (standard error):', min_value=0.0, max_value=0.396, value=0.03)
concave_points_error = st.slider('concave points (standard error):', min_value=0.0, max_value=0.053, value=0.02)
symmetry_error = st.slider('symmetry (standard error):', min_value=0.008, max_value=0.079, value=0.02)
fractal_dimension_error = st.slider('fractal dimension (standard error):', min_value=0.001, max_value=0.03, value=0.00)
worst_radius = st.slider('radius (worst):', min_value=7.93, max_value=36.04, value=16.27)
worst_texture = st.slider('texture (worst):', min_value=12.02, max_value=49.54, value=25.68)
worst_perimeter = st.slider('perimeter (worst):', min_value=50.41, max_value=251.2, value=107.26)
worst_area = st.slider('area (worst):', min_value=185.2, max_value=4254.0, value=880.58)
worst_smoothness = st.slider('smoothness (worst):', min_value=0.071, max_value=0.223, value=0.13)
worst_compactness = st.slider('compactness (worst):', min_value=0.027, max_value=1.058, value=0.25)
worst_concavity = st.slider('concavity (worst):', min_value=0.0, max_value=1.252, value=0.27)
worst_concave_points = st.slider('concave points (worst):', min_value=0.0, max_value=0.291, value=0.11)
worst_symmetry = st.slider('symmetry (worst):', min_value=0.156, max_value=0.664, value=0.29)
worst_fractal_dimension = st.slider('fractal dimension (worst):', min_value=0.055, max_value=0.208, value=0.08)

#n=212 - Malignant(=0), n=357 - Benign(=1)
labels = ['惡性（Malignant）', '良性（Benign）']
if st.button('預測'):
    X_new = [[mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness,mean_compactness,mean_concavity,mean_concave_points,mean_symmetry,mean_fractal_dimension,radius_error,texture_error,perimeter_error,area_error,smoothness_error,compactness_error,concavity_error,concave_points_error,symmetry_error,fractal_dimension_error,worst_radius,worst_texture,worst_perimeter,worst_area,worst_smoothness,worst_compactness,worst_concavity,worst_concave_points,worst_symmetry,worst_fractal_dimension]]
    X_new = scaler.transform(X_new)
    st.write('### 預測腫瘤為：', labels[clf.predict(X_new)[0]])
