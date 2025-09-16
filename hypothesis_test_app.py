import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl   # ← これが必要！
import numpy as np
from scipy import stats

# フォント設定（日本語対応）
mpl.rcParams['font.family'] = 'IPAPGothic'  # 'Noto Sans CJK JP' でも可

st.title("仮説検定シミュレーション（数学B・統計的な推測）")

# ----------------------------
# ユーザー入力
# ----------------------------
st.sidebar.header("設定")
test_type = st.sidebar.radio("検定の種類", ["両側検定", "片側検定（右側）", "片側検定（左側）"])
p0 = st.sidebar.slider("帰無仮説の確率 p₀", 0.0, 1.0, 0.5, 0.01)
n = st.sidebar.slider("標本サイズ n", 1, 200, 30)
k = st.sidebar.slider("観測された成功数 k", 0, n, n // 2)
alpha = st.sidebar.selectbox("有意水準 α", [0.01, 0.05, 0.1])

# シミュレーション設定
sim_times = st.sidebar.slider("シミュレーション回数", 100, 5000, 1000, step=100)

# ----------------------------
# 統計量の計算
# ----------------------------
E = n * p0
sigma = np.sqrt(n * p0 * (1 - p0))
z = (k - E) / sigma if sigma > 0 else 0

# 棄却域の判定関数
def reject_h0(k_obs):
    """観測値 k_obs に基づいて H0 を棄却するかどうかを返す"""
    if sigma == 0:
        return False
    z_val = (k_obs - E) / sigma
    if test_type == "両側検定":
        z_crit = norm.ppf(1 - alpha / 2)
        return (z_val <= -z_crit) or (z_val >= z_crit)
    elif test_type == "片側検定（右側）":
        z_crit = norm.ppf(1 - alpha)
        return z_val >= z_crit
    else:  # 左側
        z_crit = norm.ppf(alpha)
        return z_val <= z_crit

# ----------------------------
# 単発の判定
# ----------------------------
if test_type == "両側検定":
    z_crit = norm.ppf(1 - alpha / 2)
    reject = (z <= -z_crit) or (z >= z_crit)
elif test_type == "片側検定（右側）":
    z_crit = norm.ppf(1 - alpha)
    reject = z >= z_crit
else:  # 左側
    z_crit = norm.ppf(alpha)
    reject = z <= z_crit

st.subheader("計算結果")
st.write(f"- 期待値 E = {E:.2f}")
st.write(f"- 標準偏差 σ = {sigma:.2f}")
st.write(f"- 標準化した値 Z = {z:.2f}")
st.write(f"- 棄却域: {test_type}, 有意水準 α = {alpha}")
if reject:
    st.error("→ 帰無仮説を棄却します。")
else:
    st.success("→ 帰無仮説を棄却できません。")

# ----------------------------
# グラフ 1: 二項分布
# ----------------------------
x = np.arange(0, n+1)
binom_pmf = binom.pmf(x, n, p0)

fig, ax = plt.subplots()
ax.bar(x, binom_pmf, color="lightgray", label="二項分布 P(X=k)")
ax.bar(k, binom.pmf(k, n, p0), color="red", label=f"観測値 k={k}")
ax.set_title("二項分布による確率分布")
ax.set_xlabel("成功数 k")
ax.set_ylabel("確率")
ax.legend()
st.pyplot(fig)

# ----------------------------
# グラフ 2: 標準正規分布
# ----------------------------
x_norm = np.linspace(-4, 4, 400)
y_norm = norm.pdf(x_norm, 0, 1)

fig2, ax2 = plt.subplots()
ax2.plot(x_norm, y_norm, label="標準正規分布 N(0,1)")

# 棄却域を塗り分け
if test_type == "両側検定":
    ax2.fill_between(x_norm, 0, y_norm, where=(x_norm <= -z_crit), color="orange", alpha=0.5, label="棄却域")
    ax2.fill_between(x_norm, 0, y_norm, where=(x_norm >= z_crit), color="orange", alpha=0.5)
elif test_type == "片側検定（右側）":
    ax2.fill_between(x_norm, 0, y_norm, where=(x_norm >= z_crit), color="orange", alpha=0.5, label="棄却域")
else:  # 左側
    ax2.fill_between(x_norm, 0, y_norm, where=(x_norm <= z_crit), color="orange", alpha=0.5, label="棄却域")

# 観測値のZを縦線で
ax2.axvline(z, color="red", linestyle="--", label=f"Z = {z:.2f}")

ax2.set_title("標準正規分布による近似")
ax2.set_xlabel("標準化した値 Z")
ax2.set_ylabel("確率密度")
ax2.legend()
st.pyplot(fig2)

# ----------------------------
# シミュレーション
# ----------------------------
st.subheader("シミュレーション結果")

# 無作為抽出を sim_times 回繰り返す
samples = np.random.binomial(n, p0, size=sim_times)
rejects = [reject_h0(val) for val in samples]
reject_rate = np.mean(rejects)

st.write(f"{sim_times} 回の標本抽出で、帰無仮説が棄却された割合 = **{reject_rate:.3f}**")

# ヒストグラム表示
fig3, ax3 = plt.subplots()
ax3.hist(samples, bins=range(0, n+2), color="skyblue", alpha=0.7, edgecolor="black")
ax3.axvline(k, color="red", linestyle="--", label=f"観測値 k={k}")
ax3.set_title(f"{sim_times} 回のシミュレーション分布")
ax3.set_xlabel("成功数 k")
ax3.set_ylabel("出現回数")
ax3.legend()
st.pyplot(fig3)






