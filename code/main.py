# 축구 선수의 몸값은 정말 종합 능력치로 설명되는가
# 실행: 저장소 맨 위 폴더에서 streamlit run code/main.py
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(page_title="FC25 몸값 분석", page_icon="⚽", layout="wide")

DATA = Path("data/eafc25_players.csv")
if not DATA.exists():  # code 폴더 안에서 실행한 경우
    DATA = Path(__file__).resolve().parent.parent / "data" / "eafc25_players.csv"


def man(x):
    """유로 금액을 만 단위 글자로 바꾼다."""
    v = int(round(x / 10_000))
    if v >= 10_000:
        eok, rest = divmod(v, 10_000)
        return f"{eok}억 {rest:,}만 유로" if rest else f"{eok}억 유로"
    return f"{v:,}만 유로"


@st.cache_data
def load():
    raw = pd.read_csv(DATA)
    df = raw[raw["value_eur"] > 0].copy()  # 몸값 0은 정보 없음으로 보고 뺀다
    df["log_value"] = np.log10(df["value_eur"])
    return raw, df


@st.cache_resource
def fit_models(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    y = test["value_eur"]
    base_pred = np.full(len(test), train["value_eur"].mean())
    ma = LinearRegression().fit(train[["overall"]], train["value_eur"])
    mb = LinearRegression().fit(train[["overall"]], train["log_value"])
    mc = LinearRegression().fit(train[["overall", "age"]], train["log_value"])
    pa = ma.predict(test[["overall"]])
    pb = 10 ** mb.predict(test[["overall"]])
    pc = 10 ** mc.predict(test[["overall", "age"]])
    rows = [
        ("평균 모델(기준선)", "-", r2_score(y, base_pred), mean_absolute_error(y, base_pred)),
        ("A. overall", "원래 몸값", r2_score(y, pa), mean_absolute_error(y, pa)),
        ("B. overall", "로그 몸값", r2_score(test["log_value"], np.log10(pb)), mean_absolute_error(y, pb)),
        ("C. overall + age", "로그 몸값", r2_score(test["log_value"], np.log10(pc)), mean_absolute_error(y, pc)),
    ]
    table = pd.DataFrame(rows, columns=["모델", "예측한 값", "R²", "MAE"])
    table["R²"] = table["R²"].map(lambda r: f"{0.0 if abs(r) < 0.005 else r:.2f}")
    table["MAE"] = table["MAE"].map(man)
    result = test.copy()
    result["pred_a"] = pa
    result["pred_c"] = pc
    # 후보 모델 비교: 최종 입력(overall, age)과 로그 몸값, 같은 분할
    base_mae = mean_absolute_error(y, base_pred)
    cand_rows = []
    for name, m in [
        ("선형 회귀(최종)", LinearRegression()),
        ("k-NN 회귀(k=5)", KNeighborsRegressor(n_neighbors=5)),
        ("결정 트리 회귀(깊이 4)", DecisionTreeRegressor(max_depth=4, random_state=0)),
        ("랜덤 포레스트 회귀(300그루)", RandomForestRegressor(n_estimators=300, random_state=0)),
    ]:
        m.fit(train[["overall", "age"]], train["log_value"])
        p_log = m.predict(test[["overall", "age"]])
        mae = mean_absolute_error(y, 10 ** p_log)
        cand_rows.append((name, man(mae), f"{r2_score(test['log_value'], p_log):.2f}",
                          f"{(1 - mae / base_mae) * 100:.0f}%", "앱에서 계산"))
    # XGBoost는 브라우저(stlite)에서 돌릴 수 없어 code/check.py로 계산한 값을 적는다
    cand_rows.append(("XGBoost 회귀(기본값)", "263만 유로", "0.96", "82%", "로컬에서 계산한 값"))
    cand = pd.DataFrame(cand_rows, columns=["후보 모델", "MAE", "R²(로그 척도)", "평균 모델보다 줄어든 MAE", "계산한 곳"])
    return train, test, table, mc, result, cand


raw, df = load()
train, test, table, model_c, result, cand = fit_models(df)

st.title("⚽ 축구 선수의 몸값은 정말 종합 능력치로 설명되는가")
st.caption("FC25 선수 800명 데이터 · 20203 김도윤 · 20211 박서연 · 20219 이준호 (선생님이 만든 가상 팀의 샘플)")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["① 질문", "② 데이터 보기", "③ 기준선", "④ 모델 결과", "⑤ 해석·한계"])

with tab1:
    st.subheader("우리 팀의 질문")
    st.markdown(
        "축구 게임에서 종합 능력치(overall)가 비슷한데 몸값(value_eur)이 크게 다른 선수를 자주 봤습니다. "
        "**종합 능력치 하나로 몸값이 얼마나 설명되는지**, 설명되지 않는 부분은 무엇인지 확인합니다."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("분석한 선수", f"{len(df)}명", f"몸값 0인 {len(raw) - len(df)}명 제외", delta_color="off")
    c2.metric("방법", "선형 회귀")
    c3.metric("평가 지표", "R² · MAE")

with tab2:
    st.subheader("몸값 분포")
    scale = st.radio("보는 방법", ["원래 값(유로)", "로그 값(log10)"], horizontal=True)
    if scale.startswith("원래"):
        fig = px.histogram(df, x="value_eur", nbins=40, labels={"value_eur": "몸값(유로)"})
    else:
        fig = px.histogram(df, x="log_value", nbins=40, labels={"log_value": "몸값의 log10 (7 = 1,000만 유로, 8 = 1억 유로)"})
    fig.update_layout(yaxis_title="선수 수", bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)
    v = df["value_eur"]
    c1, c2, c3 = st.columns(3)
    c1.metric("평균", man(v.mean()))
    c2.metric("중앙값", man(v.median()))
    c3.metric("최댓값", man(v.max()))
    st.markdown(
        f"평균이 중앙값보다 크고, 평균보다 비싼 선수는 {(v > v.mean()).mean() * 100:.1f}%뿐입니다. "
        "몇 명의 아주 비싼 선수가 평균을 끌어올립니다."
    )
    st.subheader("종합 능력치와 몸값")
    fig2 = px.scatter(df, x="overall", y="value_eur", color="age", hover_name="name",
                      labels={"overall": "종합 능력치", "value_eur": "몸값(유로)", "age": "나이"},
                      color_continuous_scale="Viridis")
    st.plotly_chart(fig2, use_container_width=True)
    with st.expander("전처리: 몸값 0인 선수"):
        st.dataframe(raw[raw["value_eur"] == 0][["name", "club", "overall", "age", "value_eur"]], hide_index=True)
        st.write("9명 중 8명은 소속 구단이 비어 있습니다. 몸값을 모르는 선수로 보고 분석에서 뺐습니다.")

with tab3:
    st.subheader("기준선: 평균 모델")
    st.markdown(
        f"능력치를 전혀 보지 않고 모든 선수의 몸값을 **훈련 데이터 평균({man(train['value_eur'].mean())})**으로 예측합니다. "
        "우리 모델이 이것보다 얼마나 덜 틀리는지가 판단 기준입니다."
    )
    st.metric("평균 모델의 MAE", table.loc[0, "MAE"])
    st.caption(f"훈련 데이터 {len(train)}명으로 만들고 시험 데이터 {len(test)}명으로 평가했습니다(8:2 분할).")

with tab4:
    st.subheader("후보 모델 비교 (시험 데이터)")
    st.markdown("입력은 종합 능력치와 나이, 예측한 값은 로그 몸값으로 같게 두고 회귀 모델 다섯 개를 비교했습니다. MAE는 유로로 되돌려 계산했습니다.")
    st.dataframe(cand, hide_index=True, use_container_width=True)
    st.caption("XGBoost 회귀는 브라우저 앱에서 실행할 수 없어 code/check.py로 로컬에서 계산한 값을 적었습니다.")
    st.markdown(
        "**선형 회귀를 고른 이유**: XGBoost와 k-NN이 MAE 40만 유로를 더 줄이지만, 선형 회귀는 "
        "능력치 1점당 1.20배, 나이 한 살당 0.91배처럼 계수를 읽을 수 있습니다. "
        "수업에서 배운 방법이고, 평균 모델과의 차이(1,150만 유로)에 비하면 점수 차이가 작습니다."
    )

    st.subheader("우리 팀이 정한 입력과 예측할 값 (선형 회귀)")
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("R²는 각 모델이 예측한 값의 척도에서 계산했습니다. 서로 다른 척도의 R²는 비교하지 않고, 유로 단위의 MAE로 비교합니다.")
    which = st.radio("산점도로 볼 모델", ["A. overall", "C. overall + age"], horizontal=True)
    col = "pred_a" if which.startswith("A") else "pred_c"
    fig3 = px.scatter(result, x="value_eur", y=col, hover_name="name", hover_data=["overall", "age"],
                      labels={"value_eur": "실제 몸값(유로)", col: "예측 몸값(유로)"})
    top = float(max(result["value_eur"].max(), result[col].max()))
    fig3.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="예측 = 실제",
                              line=dict(dash="dash", color="gray")))
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("예측해 보기 (모델 C)")
    c1, c2 = st.columns(2)
    o = c1.slider("종합 능력치(overall)", int(df["overall"].min()), int(df["overall"].max()), 85)
    a = c2.slider("나이(age)", int(df["age"].min()), int(df["age"].max()), 24)
    pred = 10 ** model_c.predict(pd.DataFrame({"overall": [o], "age": [a]}))[0]
    st.metric("예측 몸값", man(pred))
    if o >= 90:
        st.warning(f"종합 능력치 90 이상인 선수는 데이터에 {(df['overall'] >= 90).sum()}명뿐입니다. 이 구간의 예측은 믿기 어렵습니다.")
    if a <= 23:
        st.info("23살 이하 선수는 시험 데이터에서 오차가 가장 컸습니다. 젊은 스타 선수는 실제보다 높게 예측되는 경향이 있습니다.")

with tab5:
    st.subheader("해석")
    st.markdown(
        "- 종합 능력치 하나로도 평균 모델보다 MAE가 크게 줄었습니다. 그래도 평균 857만 유로가 빗나갑니다.\n"
        "- 나이를 함께 넣고 몸값에 로그를 씌우자 MAE가 303만 유로로 줄었습니다. 능력치가 같으면 나이가 한 살 많을 때 예측 몸값이 약 0.91배가 됩니다.\n"
        "- 그래서 우리 팀의 답은 '종합 능력치만으로는 부족하고, 나이도 몸값에 큰 영향을 준다'입니다."
    )
    st.subheader("한계")
    st.markdown(
        "- 이 800명은 능력치 77 이상인 상위 선수입니다. 전체 프로 선수에게 그대로 적용할 수 없습니다.\n"
        "- 종합 능력치 90 이상은 8명뿐이고, 86 이상 선수의 평균 오차는 1,277만 유로입니다.\n"
        "- 계약 기간, 시장 상황, 리그 정보는 데이터에 없습니다.\n"
        "- 이 모델로 실제 이적료를 정하면 안 됩니다. 게임 회사가 매긴 몸값을 따라 한 모델입니다."
    )
