"""README·log·slides에 적은 수치를 다시 계산하는 검증 스크립트.

저장소 맨 위 폴더에서 실행한다: python3 code/check.py
앱(main.py)과 같은 전처리·분할·모델을 사용한다.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

try:  # XGBoost는 로컬 확인용. 앱(stlite)에는 넣지 않는다
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

DATA = Path("data/eafc25_players.csv")
if not DATA.exists():
    DATA = Path(__file__).resolve().parent.parent / "data" / "eafc25_players.csv"


def man(x):
    """유로 금액을 '857만 유로'처럼 만 단위로 적는다. 1억 이상은 '1억 7,450만 유로'."""
    v = int(round(x / 10_000))
    if v >= 10_000:
        eok, rest = divmod(v, 10_000)
        return f"{eok}억 {rest:,}만 유로" if rest else f"{eok}억 유로"
    return f"{v:,}만 유로"


def r2f(x):
    """R²를 소수 둘째 자리로 적는다. -0.00은 0.00으로 적는다."""
    return f"{0.0 if abs(x) < 0.005 else x:.2f}"


def group(pos):
    """첫 번째 포지션으로 공격·미드필더·수비를 나눈다."""
    if pos in ("CB", "LB", "RB", "LWB", "RWB"):
        return "수비"
    if pos in ("CDM", "CM", "CAM", "LM", "RM"):
        return "미드필더"
    return "공격"


raw = pd.read_csv(DATA)
print("== 1. 데이터 ==")
print(f"전체 행 수: {len(raw)}명, 열 수: {raw.shape[1]}개")
zero = raw[raw["value_eur"] == 0]
print(f"몸값 0인 선수: {len(zero)}명 (그중 소속 구단이 비어 있는 선수 {zero['club'].isna().sum()}명)")
print(f"소속 구단이 비어 있는 선수(전체): {raw['club'].isna().sum()}명")
print(f"골키퍼 포지션 선수: {raw['positions'].str.contains('GK').sum()}명")
print(f"제거 전 몸값 평균: {man(raw['value_eur'].mean())}")

df = raw[raw["value_eur"] > 0].copy()
df["log_value"] = np.log10(df["value_eur"])
v = df["value_eur"]
print(f"제거 후 행 수: {len(df)}명")
print(f"제거 후 몸값 평균: {man(v.mean())}, 중앙값: {man(v.median())}")
print(f"제거 후 몸값 최솟값: {man(v.min())}, 최댓값: {man(v.max())}")
print(f"평균보다 비싼 선수 비율: {(v > v.mean()).mean() * 100:.1f}%")
print(f"왜도(치우친 정도): 원래 값 {v.skew():.2f}, 로그 값 {df['log_value'].skew():.2f}")
print(f"overall 범위: {df['overall'].min()}~{df['overall'].max()}, age 범위: {df['age'].min()}~{df['age'].max()}")
print(f"overall 90 이상 선수: {(df['overall'] >= 90).sum()}명 (전체 {len(df)}명 중)")
o85 = df[df["overall"] == 85]["value_eur"]
print(f"overall 85인 선수 {len(o85)}명: 몸값 최솟값 {man(o85.min())}, 최댓값 {man(o85.max())}, 차이 {man(o85.max() - o85.min())}, {o85.max() / o85.min():.1f}배")
print(f"overall 85 이하 선수 비율: {(df['overall'] <= 85).mean() * 100:.1f}%")

train, test = train_test_split(df, test_size=0.2, random_state=42)
print(f"훈련 데이터 {len(train)}명, 시험 데이터 {len(test)}명")
print(f"overall 90 이상: 훈련 {(train['overall'] >= 90).sum()}명, 시험 {(test['overall'] >= 90).sum()}명")

y = test["value_eur"]
print("\n== 2. 기준선(평균 모델) ==")
base = np.full(len(test), train["value_eur"].mean())
print(f"훈련 데이터 평균: {man(train['value_eur'].mean())}")
print(f"평균 모델 MAE: {man(mean_absolute_error(y, base))}, R²: {r2f(r2_score(y, base))}")

print("\n== 3. 모델 A: overall 하나, 원래 몸값 ==")
ma = LinearRegression().fit(train[["overall"]], train["value_eur"])
pa = ma.predict(test[["overall"]])
print(f"R²: {r2f(r2_score(y, pa))}, MAE: {man(mean_absolute_error(y, pa))}")
print(f"overall 1 오를 때 몸값 증가: {man(ma.coef_[0])}")
print(f"평균 모델보다 MAE가 줄어든 비율: {(1 - mean_absolute_error(y, pa) / mean_absolute_error(y, base)) * 100:.0f}%")

print("\n== 4. 모델 B: overall 하나, 로그 몸값 ==")
mb = LinearRegression().fit(train[["overall"]], train["log_value"])
pb_log = mb.predict(test[["overall"]])
pb = 10 ** pb_log
print(f"로그 척도 R²: {r2f(r2_score(test['log_value'], pb_log))}")
print(f"유로로 되돌린 R²: {r2f(r2_score(y, pb))}, MAE: {man(mean_absolute_error(y, pb))}")

print("\n== 5. 모델 C: overall + age, 로그 몸값 (최종) ==")
mc = LinearRegression().fit(train[["overall", "age"]], train["log_value"])
pc_log = mc.predict(test[["overall", "age"]])
pc = 10 ** pc_log
print(f"로그 척도 R²: {r2f(r2_score(test['log_value'], pc_log))}")
print(f"유로로 되돌린 R²: {r2f(r2_score(y, pc))}, MAE: {man(mean_absolute_error(y, pc))}")
print(f"계수: overall {mc.coef_[0]:+.4f}, age {mc.coef_[1]:+.4f}")
print(f"overall 1 오르면 몸값 {10 ** mc.coef_[0]:.2f}배, 나이 1살 많으면 {10 ** mc.coef_[1]:.2f}배")
print(f"평균 모델보다 MAE가 줄어든 비율: {(1 - mean_absolute_error(y, pc) / mean_absolute_error(y, base)) * 100:.0f}%")
for o, a in [(85, 22), (85, 32)]:
    p = 10 ** mc.predict(pd.DataFrame({"overall": [o], "age": [a]}))[0]
    print(f"예시 예측 overall {o}, {a}살: {man(p)}")

print("\n== 6. 확인용: overall + age, 원래 몸값 ==")
md = LinearRegression().fit(train[["overall", "age"]], train["value_eur"])
pd_ = md.predict(test[["overall", "age"]])
print(f"R²: {r2f(r2_score(y, pd_))}, MAE: {man(mean_absolute_error(y, pd_))}")

print("\n== 7. 잔차 살펴보기 (모델 C, 시험 데이터) ==")
t = test.copy()
t["pred"] = pc
t["err"] = t["pred"] - t["value_eur"]
t["abs_err"] = t["err"].abs()
top = t.sort_values("abs_err", ascending=False).head(10)
print(f"오차 상위 10명 중 실제보다 높게 예측한 선수: {(top['err'] > 0).sum()}명")
print(f"오차 상위 10명 중 23살 이하: {(top['age'] <= 23).sum()}명")
for _, r in top.head(3).iterrows():
    print(f"  {r['name']} ({r['overall']}, {r['age']}살): 실제 {man(r['value_eur'])}, 예측 {man(r['pred'])}")
young = t[t["age"] <= 23]
old = t[t["age"] > 23]
print(f"23살 이하 {len(young)}명 평균 절대 오차: {man(young['abs_err'].mean())}")
print(f"24살 이상 {len(old)}명 평균 절대 오차: {man(old['abs_err'].mean())}")
hi = t[t["overall"] >= 86]
lo = t[t["overall"] < 86]
print(f"overall 86 이상 {len(hi)}명 평균 절대 오차: {man(hi['abs_err'].mean())}")
print(f"overall 85 이하 {len(lo)}명 평균 절대 오차: {man(lo['abs_err'].mean())}")
t["group"] = t["positions"].str.split(",").str[0].map(group)
t["rel_err"] = t["abs_err"] / t["value_eur"] * 100
for g in ["공격", "미드필더", "수비"]:
    s = t[t["group"] == g]
    print(f"포지션 {g} {len(s)}명: 평균 절대 오차 {man(s['abs_err'].mean())}, 실제 몸값 대비 {s['rel_err'].mean():.1f}%")

print("\n== 8. 후보 모델 비교 (overall + age, 로그 몸값, 같은 분할) ==")
X = ["overall", "age"]
candidates = [
    ("선형 회귀", LinearRegression()),
    ("k-NN 회귀(k=5)", KNeighborsRegressor(n_neighbors=5)),
    ("결정 트리 회귀(깊이 4)", DecisionTreeRegressor(max_depth=4, random_state=0)),
    ("랜덤 포레스트 회귀(300그루)", RandomForestRegressor(n_estimators=300, random_state=0)),
]
if XGBRegressor is not None:
    candidates.append(("XGBoost 회귀(기본값)", XGBRegressor(random_state=0)))
else:
    print("xgboost가 설치되어 있지 않아 XGBoost 회귀는 건너뜁니다.")
base_mae = mean_absolute_error(y, base)
scores = []
for name, m in candidates:
    m.fit(train[X], train["log_value"])
    p_log = m.predict(test[X])
    mae = mean_absolute_error(y, 10 ** p_log)
    scores.append((name, mae))
    print(f"{name}: MAE {man(mae)}, 로그 척도 R² {r2f(r2_score(test['log_value'], p_log))}, "
          f"평균 모델보다 MAE {(1 - mae / base_mae) * 100:.0f}% 감소")
best_name, best_mae = min(scores, key=lambda r: r[1])
lin_mae = scores[0][1]
print(f"MAE가 가장 작은 모델: {best_name} ({man(best_mae)})")
print(f"선형 회귀와의 차이: {man(lin_mae - best_mae)} (선형 회귀 MAE의 {(lin_mae - best_mae) / lin_mae * 100:.0f}%)")
print(f"참고: 평균 모델과 선형 회귀의 차이 {man(base_mae - lin_mae)}")
