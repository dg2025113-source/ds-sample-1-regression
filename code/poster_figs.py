"""v4 포스터(poster.html)에 넣을 그래프 3개를 SVG로 만들어 끼워 넣는 스크립트.

poster_figs_v3.py와 계산·그래프는 같고, 2단 균등(단 폭 260mm, poster.html과 같음)에 맞춰 그래프 폭과 높이만 다르다.
QR 2개는 poster.html의 것을 poster.html에 그대로 옮겨 두었으므로 여기서 다시 만들지 않는다.
저장소 맨 위 폴더에서 실행한다: python3 code/poster_figs_v4.py
check.py와 같은 전처리·분할·모델을 사용하므로 그래프의 값은 check.py 수치와 같다.
포스터를 만들 때만 사용하는 도구라 matplotlib·segno·xgboost가 필요하다(앱 실행에는 필요 없음).
  pip install matplotlib segno xgboost

poster.html의 <!-- FIG:이름 --> 과 <!-- /FIG:이름 --> 사이를 새 SVG로 바꾼다.
  FIG:hist     2. 데이터  분포 그래프
  FIG:bars     4. 결과    평가지표 비교 막대 그래프(기준선 + 후보 모델 전부, 선택 모델만 강조 색)
  FIG:result   4. 결과    결과 그래프(회귀: 실제-예측 산점도)
  FIG:qr_repo  제목띠     저장소 QR
  FIG:qr_app   발치       앱 QR

[다른 샘플에서 이 파일을 쓸 때]
  · 공통 부분(설정·svg_of·qr_svg·inject)은 그대로 둔다.
  · "이 샘플의 데이터와 그래프" 부분만 바꾼다. 그래프 이름(FIG:이름) 다섯 개는 그대로 사용한다.
  · 그래프 폭은 COL_W(단 폭)로 고정한다. 폭을 맞춰야 그래프 글자 20pt가 포스터에서도 20pt로 보인다.
"""
import io
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # 없는 글꼴 경고 끄기
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
import pandas as pd
import segno

# ───────────── 공통 설정(다섯 샘플이 같다) ─────────────
ROOT = Path(__file__).resolve().parent.parent
POSTER = ROOT / "poster.html"
ACCENT = "#1f3d7a"      # poster.html의 --accent와 같은 남색
GRAY = "#9aa0aa"        # 선택하지 않은 모델·나머지 점
DARK = "#4a4f57"        # 기준선
# 단 폭(인치) = (594mm - 좌우 여백 2×26mm - 단 사이 22mm) / 2 = 260mm. poster.html의 --pad-x·--gap을 바꾸면 다시 계산한다
COL_W = 260 / 25.4
# v4 그래프 높이(인치): 단 폭은 같고 높이만 정한다
HIST_H, BARS_H, RESULT_H = 3.3, 3.1, 6.2

plt.rcParams.update({
    "svg.fonttype": "none",          # 글자를 글자로 남겨 브라우저 글꼴로 그린다
    "font.family": ["Apple SD Gothic Neo", "Noto Sans KR", "sans-serif"],
    "font.size": 20,                 # 포스터의 표·그림 설명 20pt와 맞춘다
    "axes.edgecolor": "#000", "axes.labelcolor": "#000",
    "xtick.color": "#000", "ytick.color": "#000",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "svg.hashsalt": "poster",        # 다시 만들어도 같은 SVG가 나오게 한다
})


def svg_of(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)
    s = buf.getvalue()
    s = s[s.index("<svg"):]                       # XML 머리말·DOCTYPE 제거
    s = re.sub(r'\swidth="[^"]+"\sheight="[^"]+"', "", s, count=1)  # 크기는 CSS가 정한다
    s = re.sub(r"<metadata>.*?</metadata>", "", s, flags=re.S)
    return s


def qr_svg(url):
    qr = segno.make(url, error="m")
    n = qr.symbol_size(scale=1, border=2)[0]
    buf = io.BytesIO()
    qr.save(buf, kind="svg", xmldecl=False, svgns=True, scale=1, border=2, dark="#000")
    s = buf.getvalue().decode()
    s = re.sub(r'\swidth="\d+"\sheight="\d+"', f' viewBox="0 0 {n} {n}" shape-rendering="crispEdges"', s, count=1)
    return s.replace("<svg ", f'<svg role="img" aria-label="{url}" ', 1)


def inject(figs):
    html = POSTER.read_text(encoding="utf-8")
    for name, svg in figs.items():
        pat = re.compile(rf"(<!-- FIG:{name} -->).*?(<!-- /FIG:{name} -->)", re.S)
        if not pat.search(html):
            raise SystemExit(f"poster.html에 FIG:{name} 표시가 없습니다")
        html = pat.sub(lambda m: m.group(1) + svg + m.group(2), html)
    POSTER.write_text(html, encoding="utf-8")


def bar_chart(names, values, pick, base, fmt, xlabel, height=4.6, width=COL_W):
    """평가지표 비교 가로 막대. pick=선택 모델 이름, base=기준선 이름."""
    fig, ax = plt.subplots(figsize=(width, height))
    y = np.arange(len(names))[::-1]
    colors = [ACCENT if n == pick else DARK if n == base else GRAY for n in names]
    ax.barh(y, values, color=colors, height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    for t, n in zip(ax.get_yticklabels(), names):
        if n == pick:
            t.set_color(ACCENT)
            t.set_fontweight("bold")
    vmax = max(values)
    for yi, v, n in zip(y, values, names):
        ax.text(v + vmax * 0.012, yi, fmt(v), va="center", fontsize=20,
                color=ACCENT if n == pick else "#000", fontweight="bold" if n == pick else "normal")
    ax.set_xlim(0, vmax * 1.16)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{t:,.0f}" if t >= 1 or t == 0 else f"{t:.2f}"))
    ax.set_xlabel(xlabel)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return svg_of(fig)


# ───────────── 이 샘플의 데이터와 그래프(샘플마다 바꾼다) ─────────────
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

DATA = ROOT / "data" / "eafc25_players.csv"
REPO_URL = "https://github.com/greatsong/ds-sample-1-regression"
APP_URL = "https://greatsong.github.io/ds-sample-1-regression/"


def man(x):
    return f"{int(round(x / 10_000)):,}만"


# check.py와 같은 데이터 준비
raw = pd.read_csv(DATA)
df = raw[raw["value_eur"] > 0].copy()
df["log_value"] = np.log10(df["value_eur"])
train, test = train_test_split(df, test_size=0.2, random_state=42)
v = df["value_eur"]
y = test["value_eur"]
X = ["overall", "age"]

# 그래프 1(FIG:hist): 몸값 분포(791명)
fig, ax = plt.subplots(figsize=(COL_W, HIST_H))
ax.hist(v / 10_000, bins=np.arange(0, 18_001, 500), color=ACCENT, edgecolor="#fff", linewidth=0.6)
top = ax.get_ylim()[1]
for val, lab, dy in [(v.median(), f"중앙값 {man(v.median())} 유로", 0.55), (v.mean(), f"평균 {man(v.mean())} 유로", 0.85)]:
    ax.axvline(val / 10_000, color="#000", linestyle=(0, (4, 3)), linewidth=1.4)
    # 두 선이 가까워 글자는 빈 오른쪽에 두고 가는 선으로 잇는다
    ax.annotate(lab, xy=(val / 10_000, top * dy), xytext=(7_000, top * dy), va="center", fontsize=20,
                arrowprops=dict(arrowstyle="-", color="#000", linewidth=1.0, shrinkA=4, shrinkB=0))
ax.set_xlabel("몸값(만 유로)")
ax.set_ylabel("선수 수(명)")
ax.set_xticks([0, 5_000, 10_000, 15_000])
ax.set_xticklabels(["0", "5,000", "10,000", "15,000"])
svg_hist = svg_of(fig)

# 그래프 2(FIG:bars): 기준선과 후보 모델 다섯 개의 MAE(check.py 8절과 같은 모델)
base_mae = mean_absolute_error(y, np.full(len(test), train["value_eur"].mean()))
models = [
    ("선형 회귀(선택)", LinearRegression()),
    ("k-NN 회귀(k=5)", KNeighborsRegressor(n_neighbors=5)),
    ("결정 트리 회귀(깊이 4)", DecisionTreeRegressor(max_depth=4, random_state=0)),
    ("랜덤 포레스트 회귀", RandomForestRegressor(n_estimators=300, random_state=0)),
    ("XGBoost 회귀", XGBRegressor(random_state=0)),
]
names, maes = ["평균 모델(기준선)"], [base_mae]
for name, m in models:
    m.fit(train[X], train["log_value"])
    names.append(name)
    maes.append(mean_absolute_error(y, 10 ** m.predict(test[X])))
svg_bars = bar_chart(names, [m / 10_000 for m in maes], pick="선형 회귀(선택)", base="평균 모델(기준선)",
                     fmt=lambda t: f"{int(round(t)):,}만", xlabel="MAE(만 유로, 작을수록 좋음)",
                     height=BARS_H, width=COL_W)

# 그래프 3(FIG:result): 시험 데이터 159명의 실제 몸값과 최종 모델(C, 선형 회귀) 예측
mc = LinearRegression().fit(train[X], train["log_value"])
pred = 10 ** mc.predict(test[X]) / 10_000
real = y / 10_000
young = (test["age"] <= 23).to_numpy()
fig, ax = plt.subplots(figsize=(COL_W, RESULT_H))
lo, hi = 300, 20_000
ax.plot([lo, hi], [lo, hi], color="#000", linewidth=1.4, linestyle=(0, (4, 3)))
ax.scatter(real[~young], pred[~young], s=70, color=GRAY, edgecolor="none", label=f"24살 이상 {(~young).sum()}명")
ax.scatter(real[young], pred[young], s=70, color=ACCENT, edgecolor="none", label=f"23살 이하 {young.sum()}명")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ticks = [500, 1_000, 2_000, 5_000, 10_000]
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda t, _: f"{int(t):,}"))
ax.set_xlabel("실제 몸값(만 유로, 로그 눈금)")
ax.set_ylabel("예측 몸값(만 유로, 로그 눈금)")
ax.text(lo * 1.15, hi * 0.8, "점선 위: 실제보다 높게 예측", va="top", fontsize=20)
ax.legend(loc="lower right", frameon=False, fontsize=20, handletextpad=0.2, borderaxespad=0.1)
svg_result = svg_of(fig)

inject({"hist": svg_hist, "bars": svg_bars, "result": svg_result})
print("MAE(만 유로):", ", ".join(f"{n} {m / 10_000:,.0f}" for n, m in zip(names, maes)))
print(f"poster.html 갱신: 그래프 3개 (시험 데이터 {len(test)}명, 23살 이하 {young.sum()}명)")
