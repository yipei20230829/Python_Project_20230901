# https://docs.streamlit.io/library/cheatsheet
# streamlit run app.py
import streamlit as st
import numpy as np 
import joblib
import base64

#設定函數get_image_html，使用讀取圖片，以便呼叫函數直接讀取即可
def get_image_html(page_name, file_name):
    with open(file_name, "rb") as f:
        contents = f.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    return f'<a href="{page_name}"><img src="data:image/png;base64,{data_url}" style="width:300px"></a>'

#呼叫函數get_image_html
data_url = get_image_html("分類_乳癌腫瘤", "./Breast_Cancer.png")
data_url_2 = get_image_html("分類_企鵝品種", "./penguins.png")
data_url_3 = get_image_html("迴歸_波士頓房價", "./Boston_houseprice.png")
data_url_4 = get_image_html("迴歸_計程車小費", "./taxi.png")
data_url_5 = get_image_html("手寫_辨識數字", "./Write_Numbers.png")
data_url_6 = get_image_html("手寫_辨識英文字母", "./letter_write.png")

st.set_page_config(
    page_title="我的學習歷程",
    page_icon="👋",
)

st.title('Machine Learning 學習歷程')   

tab1, tab2 , tab3 , tab4 , tab5 , tab6= st.tabs(["Breast Cancer","Penguins","Boston House Price","Taxi Tips","Prediction Number","Prediction English Alphabet"])
with tab1:
    # url must be external url instead of local file
    # st.markdown(f"### [![分類]({url})](分類)")
    st.markdown('### [(分類)乳房腫瘤辨識](分類_乳癌腫瘤)')
    st.markdown('''
    ##### 特徵(X):
        - radius (mean)
        - texture (mean)
        - perimeter (mean)
        - area (mean)
        - smoothness (mean)
        - compactness (mean)
        - concavity (mean)
        - concave points (mean)
        - symmetry (mean)
        - fractal dimension (mean)
        - radius (standard error)
        - texture (standard error)
        - perimeter (standard error)
        - area (standard error)
        - smoothness (standard error)
        - compactness (standard error)
        - concavity (standard error)
        - concave points (standard error)
        - symmetry (standard error)
        - fractal dimension (standard error)
        - radius (worst)
        - texture (worst)
        - perimeter (worst)
        - area (worst)
        - smoothness (worst)
        - compactness (worst)
        - concavity (worst)
        - concave points (worst)
        - symmetry (worst)
        - fractal dimension (worst)
    ##### 預測類別(Class):
        - 惡性（Malignant）
        - 良性（Benign）
        ''')
    st.markdown(data_url, unsafe_allow_html=True)
with tab2:
    # url must be external url instead of local file
    # st.markdown(f"### [![分類]({url})](分類)")
    st.markdown('### [(分類)企鵝品種辨識](分類_企鵝品種)')
    st.markdown('''
    ##### 特徵(X):
        - 島嶼
        - 嘴巴長度
        - 嘴巴寬度
        - 翅膀長度
        - 體重
        - 性別
    ##### 預測類別(Class):
        - Adelie
        - Chinstrap
        - Gentoo
        ''')
    # st.image('iris.png')效果不佳
    st.markdown(data_url_2, unsafe_allow_html=True)
with tab3:
    st.markdown('### [(迴歸)Boston 房價預測](迴歸_波士頓房價)')
    st.markdown('''
    ##### 特徵(X):
        - 犯罪率
        - 大坪數房屋比例
        - 非零售業的營業面積比例
        - 是否靠近河岸
        - 一氧化氮濃度
        - 平均房間數
        - 屋齡(1940年前建造比例)
        - 與商業區距離
        - 與高速公路距離
        - 地價稅
        - 師生比例
        - 黑人比例(Bk — 0.63)²
        - 低下階級的比例
    ##### 目標：預測房價
        ''')

    # st.image('taxi.png')
    st.markdown(data_url_3, unsafe_allow_html=True)


    
with tab4:
    st.markdown('### [(迴歸)計程車小費預測](迴歸_計程車小費)')
    st.markdown('''
    ##### 特徵(X):
        - 車費
        - 性別
        - 吸菸
        - 星期
        - 時間
        - 同行人數
    ##### 目標：預測小費金額
        ''')

    # st.image('taxi.png')
    st.markdown(data_url_4, unsafe_allow_html=True)
with tab5:
    st.markdown('### [(手寫)數字0~9辨識](手寫_辨識數字)')


    # st.image('taxi.png')
    st.markdown(data_url_5, unsafe_allow_html=True)
with tab6:
    st.markdown('### [(手寫)英文字母辨識](手寫_辨識英文字母)')


    # st.image('taxi.png')
    st.markdown(data_url_6, unsafe_allow_html=True)
