"""포스터(poster.html)에 넣을 그래프 2개와 QR 코드 2개를 SVG로 만들어 끼워 넣는 스크립트.

저장소 맨 위 폴더에서 실행한다: python3 code/poster_figs.py
check.py와 같은 전처리·분할·모델을 사용하므로 그래프의 값은 check.py 수치와 같다.
포스터를 만들 때만 사용하는 도구라 matplotlib과 segno가 필요하다(앱 실행에는 필요 없음).
  pip install matplotlib segno

poster.html의 <!-- FIG:이름 --> 과 <!-- /FIG:이름 --> 사이를 새 SVG로 바꾼다.
"""
import io
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # 없는 글꼴 경고 끄기
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator
import numpy as np
import pandas as pd
import segno
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
POSTER = ROOT / "poster.html"
DATA = ROOT / "data" / "eafc25_players.csv"

ACCENT = "#1d5fbf"      # poster.html의 --accent와 같은 색
GRAY = "#8a909c"
# 그래프 폭(인치) = 그래프가 들어가는 열의 폭. 폭을 맞춰야 그래프 글자 20pt가 포스터에서도 20pt로 보인다.
# 열 폭은 render_poster.py가 출력하는 col 폭(px)을 96으로 나눈 값(poster.html의 --cols를 바꾸면 다시 맞춘다)
HIST_W = 6.3            # 1열
RESULT_W = 7.37         # 3열
REPO_URL = "https://github.com/greatsong/ds-sample-1-regression"
APP_URL = "https://greatsong.github.io/ds-sample-1-regression/"

plt.rcParams.update({
    "svg.fonttype": "none",          # 글자를 글자로 남겨 브라우저 글꼴로 그린다
    "font.family": ["Apple SD Gothic Neo", "Noto Sans KR", "sans-serif"],
    "font.size": 20,                 # 포스터 본문 최소 20pt와 맞춘다
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


def man(x):
    return f"{int(round(x / 10_000)):,}만"


# check.py와 같은 데이터 준비
raw = pd.read_csv(DATA)
df = raw[raw["value_eur"] > 0].copy()
df["log_value"] = np.log10(df["value_eur"])
train, test = train_test_split(df, test_size=0.2, random_state=42)
v = df["value_eur"]

# 그래프 1: 몸값 분포(791명)
fig, ax = plt.subplots(figsize=(HIST_W, 5.6))
ax.hist(v / 10_000, bins=np.arange(0, 18_001, 500), color=ACCENT, edgecolor="#fff", linewidth=0.6)
top = ax.get_ylim()[1]
for val, lab, dy in [(v.median(), f"중앙값 {man(v.median())}", 0.62), (v.mean(), f"평균 {man(v.mean())}", 0.86)]:
    ax.axvline(val / 10_000, color="#000", linestyle=(0, (4, 3)), linewidth=1.4)
    # 두 선이 가까워 글자는 빈 오른쪽에 두고 가는 선으로 잇는다
    ax.annotate(lab, xy=(val / 10_000, top * dy), xytext=(6_500, top * dy), va="center", fontsize=20,
                arrowprops=dict(arrowstyle="-", color="#000", linewidth=1.0, shrinkA=4, shrinkB=0))
ax.set_xlabel("몸값(만 유로)")
ax.set_ylabel("선수 수(명)")
ax.set_xticks([0, 5_000, 10_000, 15_000])
ax.set_xticklabels(["0", "5,000", "10,000", "15,000"])
svg_hist = svg_of(fig)

# 그래프 2: 시험 데이터 159명의 실제 몸값과 최종 모델(C, 선형 회귀) 예측
mc = LinearRegression().fit(train[["overall", "age"]], train["log_value"])
pred = 10 ** mc.predict(test[["overall", "age"]]) / 10_000
real = test["value_eur"] / 10_000
young = (test["age"] <= 23).to_numpy()
fig, ax = plt.subplots(figsize=(RESULT_W, 5.2))
lo, hi = 300, 20_000
ax.plot([lo, hi], [lo, hi], color="#000", linewidth=1.4, linestyle=(0, (4, 3)))
ax.scatter(real[~young], pred[~young], s=46, color=GRAY, edgecolor="none", label=f"24살 이상 {(~young).sum()}명")
ax.scatter(real[young], pred[young], s=46, color=ACCENT, edgecolor="none", label=f"23살 이하 {young.sum()}명")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ticks = [500, 1_000, 2_000, 5_000, 10_000]
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda t, _: f"{int(t):,}"))
ax.set_xlabel("실제 몸값(만 유로, 로그 눈금)")
ax.set_ylabel("예측 몸값(만 유로)")
ax.text(lo * 1.15, hi * 0.8, "점선 위: 실제보다 높게 예측", va="top", fontsize=20)
ax.legend(loc="lower right", frameon=False, fontsize=20, handletextpad=0.2, borderaxespad=0.1)
svg_result = svg_of(fig)


def qr_svg(url):
    qr = segno.make(url, error="m")
    n = qr.symbol_size(scale=1, border=2)[0]
    buf = io.BytesIO()
    qr.save(buf, kind="svg", xmldecl=False, svgns=True, scale=1, border=2, dark="#000")
    s = buf.getvalue().decode()
    s = re.sub(r'\swidth="\d+"\sheight="\d+"', f' viewBox="0 0 {n} {n}" shape-rendering="crispEdges"', s, count=1)
    return s.replace("<svg ", f'<svg role="img" aria-label="{url}" ', 1)


figs = {"hist": svg_hist, "result": svg_result, "qr_repo": qr_svg(REPO_URL), "qr_app": qr_svg(APP_URL)}
html = POSTER.read_text(encoding="utf-8")
for name, svg in figs.items():
    pat = re.compile(rf"(<!-- FIG:{name} -->).*?(<!-- /FIG:{name} -->)", re.S)
    if not pat.search(html):
        raise SystemExit(f"poster.html에 FIG:{name} 표시가 없습니다")
    html = pat.sub(lambda m: m.group(1) + svg + m.group(2), html)
POSTER.write_text(html, encoding="utf-8")
print(f"poster.html 갱신: 그래프 2개, QR 2개 (시험 데이터 {len(test)}명, 23살 이하 {young.sum()}명)")
