import math
import numpy as np
from scipy.stats import t as tdist
import streamlit as st

from app import one_sample_t


st.title("One-sample t-test Calculator")

mode = st.radio("Input type", ["Raw data", "Summary statistics"], index=0)

def parse_numbers_from_text(text: str):
    if not text:
        return []
    parts = [p.strip() for p in text.replace(',', ' ').split()]
    nums = []
    for p in parts:
        if p == '':
            continue
        try:
            nums.append(float(p))
        except ValueError:
            continue
    return nums


if mode == "Raw data":
    st.write("Provide raw numeric observations (paste or upload).")
    uploaded = st.file_uploader("Upload CSV or TXT (one column or plain numbers)", type=["csv", "txt"])
    text_input = st.text_area("Or paste numbers here (comma/newline separated)", height=120)

    data = []
    if uploaded is not None:
        try:
            uploaded.seek(0)
            raw = uploaded.read().decode('utf-8')
            data = parse_numbers_from_text(raw)
        except Exception:
            data = []
    if text_input and not data:
        data = parse_numbers_from_text(text_input)

    mu0 = st.number_input("Null mean (mu0)", value=0.0)
    alpha = st.number_input("Alpha (significance level)", value=0.05, min_value=0.0, max_value=1.0, format="%.4f")
    alternative = st.selectbox("Alternative hypothesis", options=["two-sided", "greater", "less"])

    if not data:
        st.info("No numeric data provided yet.")
    elif len(data) < 2:
        st.warning("Need at least two observations to run the t-test.")
    else:
        st.write(f"Loaded {len(data)} observations.")
        if st.button("Compute t-test"):
            t_stat, p_value, decision = one_sample_t(data, mu0, alpha=alpha, alternative=alternative)
            st.subheader("Results")
            st.write("t statistic:", float(t_stat))
            st.write("p-value:", float(p_value))
            st.write("Decision:", decision)

elif mode == "Summary statistics":
    st.write("Provide sample summary statistics: sample mean, sample std (sample, not population), and sample size.")
    sample_mean = st.number_input("Sample mean", value=0.0, format="%.6f")
    sample_std = st.number_input("Sample standard deviation (s)", value=1.0, format="%.6f", min_value=0.0)
    n = st.number_input("Sample size (n)", value=10, min_value=2, format="%d")
    mu0 = st.number_input("Null mean (mu0)", value=0.0, format="%.6f")
    alpha = st.number_input("Alpha (significance level)", value=0.05, min_value=0.0, max_value=1.0, format="%.4f")
    alternative = st.selectbox("Alternative hypothesis", options=["two-sided", "greater", "less"])

    if st.button("Compute from summary"):
        df = int(n) - 1
        if df < 1 or sample_std <= 0 or n < 2:
            st.error("Invalid summary inputs: need n>=2 and s>0.")
        else:
            t_stat = (sample_mean - mu0) / (sample_std / math.sqrt(n))
            if alternative == 'two-sided':
                p_value = 2 * (1 - tdist.cdf(abs(t_stat), df))
            elif alternative == 'greater':
                p_value = 1 - tdist.cdf(t_stat, df)
            else:
                p_value = tdist.cdf(t_stat, df)
            decision = "Reject H0" if p_value < alpha else "Fail to Reject H0"
            st.subheader("Results")
            st.write("t statistic:", float(t_stat))
            st.write("p-value:", float(p_value))
            st.write("Decision:", decision)

            # optional CI for two-sided
            if alternative == 'two-sided':
                crit = tdist.ppf(1 - alpha/2, df)
                se = sample_std / math.sqrt(n)
                ci_low = sample_mean - crit * se
                ci_high = sample_mean + crit * se
                st.write(f"{100*(1-alpha):.1f}% CI for mean: ({ci_low:.6f}, {ci_high:.6f})")

st.write("---")
st.write("Calculator focuses on a standard one-sample t-test. For raw-data mode the exact test uses your uploaded/pasted observations.")
