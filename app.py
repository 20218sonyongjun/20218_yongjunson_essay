import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import streamlit as st

# 페이지 기본 설정
st.set_page_config(
    page_title="산업별 주가지수 변동에 대한 회귀분석", layout="wide"
)

# 1. 제목 및 개요, 주제 선정 동기
st.title("산업별 주가지수 변동에 대한 회귀분석")

st.markdown("### < 개요 >", unsafe_allow_html=True)
st.info(
    "본 프로젝트는 2000년부터 2025년까지의 장기 산업별 주가지수 데이터를 바탕으로, 각 산업의 과거 트렌드를 탐색하고 선형회귀 모델을 활용하여 미래 주가지수 변동 양상을 분석·예측하는 데이터 과학 프로젝트입니다."
)

st.markdown("### < 주제 선정 동기 >", unsafe_allow_html=True)
st.write(
    "평소에 신문기사에서 주가지수 변동폭이 크다는 것을 느껴서 앞으로 미래에는 주가지수가 어떻게 변화할 것인가를 과거 2000년~2025년까지의 데이터를 이용해 회귀분석을 하여 알아보고 싶기 때문이다."
)

st.markdown("---")


# 데이터 로드 함수 (실제 업로드된 CSV 파일 연동 - cp949 인코딩 적용)
@st.cache_data
def load_data():
  file_name = "코스피_산업별_주가지수_20261110114354.csv"
  try:
    # cp949 인코딩으로 CSV 읽기
    df_raw = pd.read_csv(file_name, encoding="cp949")

    # 원본 구조: 행(지수종류별(1))과 열(2000~2025년)을 시계열 분석에 맞게 melt (변환)
    id_col = df_raw.columns[0]  # '지수종류별(1)'
    df_melted = df_raw.melt(
        id_vars=[id_col], var_name="Year", value_name="Index"
    )
    df_melted.rename(
        columns={id_col: "Industry", "Year": "Year", "Index": "Index"},
        inplace=True,
    )

    # 데이터 타입 정돈 (Year는 정수, Index는 숫자로 변환)
    df_melted["Year"] = df_melted["Year"].astype(int)
    df_melted["Index"] = pd.to_numeric(
        df_melted["Index"].astype(str).str.replace(",", ""), errors="coerce"
    )

    return df_melted
  except Exception as e:
    st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
    return pd.DataFrame()


df_melted = load_data()

if not df_melted.empty:
  # 산업 품목 리스트 추출
  industries = sorted(df_melted["Industry"].unique().tolist())

  # 페이지 섹션 나누기 (라디오 버튼 활용)
  menu = st.radio(
      "페이지 이동",
      [
          "1. 산업별 주가지수 (2000년~2025년)",
          "2. 품목별 회귀분석 및 미래 예측",
      ],
      horizontal=True,
  )

  st.markdown("---")

  # -------------------------------------------------------------
  # 페이지 1: 산업별 주가지수 (2000년~2025년) 비교
  # -------------------------------------------------------------
  if menu == "1. 산업별 주가지수 (2000년~2025년)":
    st.subheader("📊 산업별 주가지수 시계열 비교 (2000년~2025년)")
    st.markdown(
        "아래 검색창(다중 선택)에서 비교하고 싶은 산업 품목들을 선택해"
        " 주세요."
    )

    # 초기 화면은 빈칸(default=[])으로 설정하여 그래프를 비워둠
    selected_items = st.multiselect(
        "조사할 품목을 선택하세요 (여러 개 선택 가능)",
        options=industries,
        default=[],
    )

    if not selected_items:
      st.warning(
          "⚠️ 선택된 품목이 없습니다. 비교할 산업 품목을 위 검색창에서"
          " 선택해주세요."
      )
    else:
      # 선택된 품목들만 필터링
      df_filtered = df_melted[df_melted["Industry"].isin(selected_items)]

      # 다중 꺾은선 그래프 생성 (Plotly)
      fig = px.line(
          df_filtered,
          x="Year",
          y="Index",
          color="Industry",
          markers=True,
          title="선택한 산업별 주가지수 꺾은선 그래프 비교",
          labels={"Index": "주가지수", "Year": "연도", "Industry": "산업 품목"},
      )
      fig.update_layout(
          hovermode="x unified",
          template="plotly_white",
          height=500,
          xaxis=dict(dtick=2),
      )
      st.plotly_chart(fig, use_container_width=True)

  # -------------------------------------------------------------
  # 페이지 2: 품목별 추세선과 미래 예측 회귀분석
  # -------------------------------------------------------------
  elif menu == "2. 품목별 회귀분석 및 미래 예측":
    st.subheader("📈 품목별 추세선 및 미래 주가지수 회귀분석")
    st.markdown(
        "회귀분석을 진행할 단일 품목을 선택하여 과거 추세선과 미래 예측 양상을"
        " 확인합니다."
    )

    target_item = st.selectbox(
        "회귀분석 대상 품목 선택", options=industries
    )

    if target_item:
      # 데이터 추출 및 결측치 제거
      df_item = (
          df_melted[df_melted["Industry"] == target_item].dropna().copy()
      )
      df_item = df_item.sort_values("Year")

      X = df_item[["Year"]]  # 독립변수: 연도
      y = df_item["Index"]  # 종속변수: 주가지수

      # 선형회귀(Linear Regression) 모델 학습
      model = LinearRegression()
      model.fit(X, y)
      df_item["Trend"] = model.predict(X)

      # 미래 예측 연도 설정 (예: 2026년 ~ 2035년까지 10년 연장)
      last_year = int(df_item["Year"].max())
      future_years = np.arange(last_year + 1, last_year + 11)
      future_X = pd.DataFrame({"Year": future_years})
      future_pred = model.predict(future_X)

      df_future = pd.DataFrame(
          {"Year": future_years, "Index": future_pred, "Industry": target_item}
      )

      # Plotly 시각화 구성
      fig = go.Figure()

      # 1. 실제 과거 주가지수 그래프
      fig.add_trace(
          go.Scatter(
              x=df_item["Year"],
              y=df_item["Index"],
              mode="lines+markers",
              name="실제 주가지수 (2000~2025)",
              line=dict(color="#1f77b4", width=2),
          )
      )
      # 2. 선형회귀 추세선
      fig.add_trace(
          go.Scatter(
              x=df_item["Year"],
              y=df_item["Trend"],
              mode="lines",
              name="선형회귀 추세선",
              line=dict(color="#ff7f0e", width=2, dash="dash"),
          )
      )
      # 3. 미래 예측선 (2026~2035)
      fig.add_trace(
          go.Scatter(
              x=df_future["Year"],
              y=df_future["Index"],
              mode="lines+markers",
              name="미래 예측선 (회귀 연장)",
              line=dict(color="#2ca02c", width=2, dash="dot"),
          )
      )

      fig.update_layout(
          title=f"[{target_item}] 주가지수 회귀분석 및 미래 변동 양상 예측",
          xaxis_title="연도",
          yaxis_title="주가지수",
          template="plotly_white",
          height=500,
          xaxis=dict(dtick=2),
      )
      st.plotly_chart(fig, use_container_width=True)

      # 통계 지표(결정계수 및 회귀 계수) 출력
      r2 = model.score(X, y)
      slope = model.coef_[0]
      intercept = model.intercept_

      c1, c2, c3 = st.columns(3)
      c1.metric("결정계수 ($R^2$)", f"{r2:.4f}")
      c2.metric("회귀식 기울기 (계수)", f"{slope:.4f}")
      c3.metric("회귀식 절편", f"{intercept:.4f}")

      st.markdown(
          "#### 💡 회귀분석 결과 통계적 해석"
          f"\n- 선택한 **{target_item}** 품목의 선형회귀 모델 결정계수($R^2$)는"
          f" `{{r2:.4f}}`입니다. (1에 가까울수록 선형 모델의 설명력이 높음을"
          " 의미합니다)"
          f"\n- 연간 주가지수 변화율(기울기)은 `{{slope:.4f}}`로,"
          f" `{{'상승(양의 방향)' if slope > 0 else '하락(음의 방향)'}}` 추세를"
          " 보이고 있습니다."
          f"\n- 도출된 회귀 방정식: $\\text{{주가지수}} = {{slope:.4f}} \\times"
          f" \\text{{연도}} + ({intercept:.4f})$"
      )
