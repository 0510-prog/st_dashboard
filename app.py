import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# 페이지 기본 설정
st.set_page_config(
    page_title="네이버 쇼핑 데이터 분석 대시보드",
    page_icon="📊",
    layout="wide"
)

# 커스텀 CSS (UI 개선)
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        header {visibility: hidden;}
    }
    h1 {
        color: #2e3b4e;
        border-bottom: 2px solid #edeff2;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 5px 5px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 데이터 로드 환경 설정
# 데이터 로드 환경 설정
# 스크립트 파일(app.py)이 있는 위치를 기준으로 data 폴더 경로 설정 (로컬/배포 환경 호환성 확보)
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(current_dir, "data")

def get_latest_files():
    """데이터 디렉토리에서 사용 가능한 파일 목록과 키워드를 추출"""
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    keywords = set()
    for f in files:
        basename = os.path.basename(f)
        keyword = basename.split('_')[0]
        keywords.add(keyword)
    return sorted(list(keywords)), files

available_keywords, all_files = get_latest_files()

# --- 사이드바 구성 ---
st.sidebar.title("🔍 분석 설정")
selected_keywords = st.sidebar.multiselect(
    "비교할 키워드를 선택하세요",
    options=available_keywords,
    default=available_keywords[:2] if len(available_keywords) >= 2 else available_keywords
)

st.sidebar.markdown("---")
st.sidebar.info("네이버 API를 통해 수집된 최근 데이터를 기반으로 분석을 수행합니다.")

# --- 데이터 로딩 및 전처리 로직 ---
@st.cache_data
def load_and_preprocess(keywords):
    trend_dfs = []
    shop_dfs = []
    blog_dfs = []
    
    for kw in keywords:
        # 트렌드 데이터
        trend_file = glob.glob(os.path.join(DATA_DIR, f"{kw}_쇼핑트렌드_*.csv"))
        if trend_file:
            df = pd.read_csv(trend_file[0])
            df['period'] = pd.to_datetime(df['period'])
            df['keyword'] = kw
            trend_dfs.append(df)
            
        # 쇼핑 데이터
        shop_file = glob.glob(os.path.join(DATA_DIR, f"{kw}_네이버쇼핑_*.csv"))
        if shop_file:
            df = pd.read_csv(shop_file[0])
            df['keyword'] = kw
            df['lprice'] = pd.to_numeric(df['lprice'], errors='coerce')
            shop_dfs.append(df)
            
        # 블로그 데이터
        blog_file = glob.glob(os.path.join(DATA_DIR, f"{kw}_블로그게시물_*.csv"))
        if blog_file:
            df = pd.read_csv(blog_file[0])
            df['keyword'] = kw
            blog_dfs.append(df)
            
    return (
        pd.concat(trend_dfs, ignore_index=True) if trend_dfs else pd.DataFrame(),
        pd.concat(shop_dfs, ignore_index=True) if shop_dfs else pd.DataFrame(),
        pd.concat(blog_dfs, ignore_index=True) if blog_dfs else pd.DataFrame()
    )

if not selected_keywords:
    st.warning("분석을 위해 최소 하나 이상의 키워드를 선택해주세요.")
    st.stop()

trend_all, shop_all, blog_all = load_and_preprocess(selected_keywords)

# --- 메인 화면 ---
st.title("📊 네이버 쇼핑 및 콘텐츠 연합 분석 대시보드")

tab1, tab2, tab3 = st.tabs(["📉 쇼핑 트렌드 분석", "🛍️ 상품/가격 EDA", "📝 블로그 콘텐츠 분석"])

# --- Tab 1: 트렌드 분석 ---
with tab1:
    st.subheader("키워드별 클릭 지수 변화")
    
    if not trend_all.empty:
        # 1. 시계열 라인 차트
        fig_line = px.line(
            trend_all, x='period', y='ratio', color='keyword',
            title="날짜별 클릭 지수 추이 (상대값)",
            labels={'period': '날짜', 'ratio': '클릭 지수', 'keyword': '키워드'},
            template="plotly_white"
        )
        st.plotly_chart(fig_line, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 클릭 지수 분포 (박스플롯)")
            fig_box = px.box(
                trend_all, x='keyword', y='ratio', color='keyword',
                points="all", title="키워드별 관심도 편차 확인",
                labels={'keyword': '키워드', 'ratio': '클릭 지수'}
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
        with col2:
            st.markdown("#### 주요 통계 요약")
            stats_df = trend_all.groupby('keyword')['ratio'].describe().reset_index()
            st.dataframe(stats_df.style.highlight_max(axis=0, subset=['mean', 'max']), use_container_width=True)

        st.markdown("#### 월별 평균 클릭 지수 추이")
        trend_all['month'] = trend_all['period'].dt.strftime('%Y-%m')
        monthly_pivot = trend_all.pivot_table(index='month', columns='keyword', values='ratio', aggfunc='mean')
        st.table(monthly_pivot.style.format("{:.2f}"))
    else:
        st.info("트렌드 데이터가 존재하지 않습니다.")

# --- Tab 2: 상품/가격 분석 ---
with tab2:
    st.subheader("네이버 쇼핑 상품 데이터 분석")
    
    if not shop_all.empty:
        col3, col4 = st.columns([3, 2])
        
        with col3:
            st.markdown("#### 키워드별 가격 분포 (히스토그램)")
            fig_hist = px.histogram(
                shop_all, x='lprice', color='keyword', barmode='overlay',
                marginal="rug", title="상품 가격 밀집 구간 분석",
                labels={'lprice': '가격(원)', 'keyword': '키워드'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col4:
            st.markdown("#### 인기 브랜드 TOP 10 (빈도)")
            brand_counts = shop_all.groupby(['keyword', 'brand']).size().reset_index(name='count')
            brand_counts = brand_counts.sort_values(['keyword', 'count'], ascending=[True, False])
            # 각 키워드별 상위 5개씩만 보여주기
            top_brands = brand_counts.groupby('keyword').head(5)
            st.dataframe(top_brands, use_container_width=True)

        st.markdown("#### 상품별 가격 분포 산점도")
        fig_scatter = px.scatter(
            shop_all, x=shop_all.index, y='lprice', color='keyword',
            hover_data=['title', 'mallName'],
            title="전체 수집 상품 가격 포지셔닝",
            labels={'index': '상품 순서', 'lprice': '가격(원)'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        col5, col6 = st.columns(2)
        with col5:
            st.markdown("#### 주요 판매처(Mall) 분포")
            mall_cross = pd.crosstab(shop_all['mallName'], shop_all['keyword']).sort_values(by=selected_keywords[0], ascending=False).head(10)
            st.dataframe(mall_cross, use_container_width=True)
            
        with col6:
            st.markdown("#### 주요 상품 리스트 (최저가 상위)")
            top_products = shop_all.sort_values('lprice').head(10)[['title', 'lprice', 'mallName', 'keyword']]
            st.dataframe(top_products, use_container_width=True)
    else:
        st.info("상품 검색 데이터가 존재하지 않습니다.")

# --- Tab 3: 블로그 분석 ---
with tab3:
    st.subheader("블로그 콘텐츠 검색 키워드 분석")
    
    if not blog_all.empty:
        for kw in selected_keywords:
            st.markdown(f"#### [{kw}] 관련 주요 핵심 어휘 (TF-IDF)")
            kw_blog = blog_all[blog_all['keyword'] == kw]['title'].fillna('')
            
            if not kw_blog.empty:
                vectorizer = TfidfVectorizer(max_features=20)
                tfidf_matrix = vectorizer.fit_transform(kw_blog)
                feature_names = vectorizer.get_feature_names_out()
                sums = tfidf_matrix.sum(axis=0)
                
                ranking = pd.DataFrame([
                    {'단어': name, 'TF-IDF': sums[0, i]} for i, name in enumerate(feature_names)
                ]).sort_values('TF-IDF', ascending=False)
                
                fig_tfidf = px.bar(
                    ranking, x='TF-IDF', y='단어', orientation='h',
                    title=f"{kw} 블로그 핵심 키워드",
                    color='TF-IDF', color_continuous_scale='Viridis'
                )
                fig_tfidf.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_tfidf, use_container_width=True)
                
                with st.expander(f"{kw} 블로그 게시물 원문 보기"):
                    st.dataframe(blog_all[blog_all['keyword'] == kw][['title', 'link']].head(10))
            else:
                st.write(f"{kw} 에 대한 블로그 데이터가 부족합니다.")
    else:
        st.info("블로그 검색 데이터가 존재하지 않습니다.")

st.markdown("---")
st.caption("Produced by Antigravity AI Dashboard System")
