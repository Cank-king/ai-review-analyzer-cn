"""AI 电商评论分析器 Streamlit 应用。"""

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from analyzer import analyze_reviews, generate_ai_business_report, generate_business_report


st.set_page_config(page_title="商家评论诊断台", page_icon="🧭", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1200px; padding-top: 2rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#17324d,#256b70); color:white; margin-bottom:1.2rem;}
.hero h1 {margin:0 0 .35rem 0; font-size:2rem;}
.hero p {margin:0; opacity:.86; font-size:1rem;}
.report-card {padding:1rem 1.15rem; border:1px solid #e6eaf0; border-radius:14px; background:#fff; min-height:125px;}
.report-card h4 {margin:0 0 .5rem 0; color:#17324d;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🧭 商家评论诊断台</h1><p>把消费者反馈，整理成可以直接执行的经营决策。</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("数据设置")
    uploaded_file = st.file_uploader("上传评论 CSV", type=["csv"], help="建议包含评论内容和评分两列。")
    st.caption("支持中文 CSV；页面不会展示或上传你的 API Key。")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"已加载 {len(df):,} 条评论")
    except Exception as exc:
        st.error(f"CSV 读取失败：{exc}")
        st.stop()
else:
    df = pd.read_csv(Path(__file__).with_name("sample_reviews.csv"))
    st.info("当前使用示例数据演示；从左侧上传 CSV 可分析自己的评论。")

if df.empty:
    st.warning("CSV 文件没有数据。")
    st.stop()

columns = list(df.columns)
default_text = columns.index("评论内容") if "评论内容" in columns else 0
default_rating = columns.index("评分") if "评分" in columns else min(1, len(columns) - 1)
left, right = st.columns(2)
with left:
    text_column = st.selectbox("评论内容列", columns, index=default_text)
with right:
    rating_column = st.selectbox("评分列", columns, index=default_rating)

try:
    result = analyze_reviews(df, text_column, rating_column)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

positive_rate = result["positive"] / result["total"] * 100 if result["total"] else 0
negative_rate = result["negative"] / result["total"] * 100 if result["total"] else 0
st.subheader("经营概览")
metrics = st.columns(4)
metrics[0].metric("评论总数", f'{result["total"]:,}')
metrics[1].metric("好评率", f"{positive_rate:.1f}%")
metrics[2].metric("差评率", f"{negative_rate:.1f}%")
metrics[3].metric("待关注评论", f'{result["negative"]:,}')

tab_overview, tab_report, tab_data = st.tabs(["📊 评论洞察", "🧠 AI 商家诊断", "📁 原始数据"])

with tab_overview:
    st.caption("基于评分、关键词和规则识别出的主要反馈方向。")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("高频关键词")
        if result["keywords"]:
            st.dataframe(pd.DataFrame(result["keywords"], columns=["关键词", "出现次数"]), hide_index=True, use_container_width=True)
        else:
            st.write("暂未提取到有效关键词。")
    with col2:
        st.subheader("评分分布")
        st.bar_chart(pd.Series({"好评": result["positive"], "中性": result["neutral"], "差评": result["negative"]}))
    col_good, col_bad = st.columns(2)
    with col_good:
        st.subheader("常见好评原因")
        for reason, count in result["positive_reasons"] or [("暂无明显好评原因", "")]:
            st.write(f"✅ {reason}" + (f"（{count}次）" if count != "" else ""))
    with col_bad:
        st.subheader("常见差评原因")
        for reason, count in result["negative_reasons"] or [("暂无明显差评原因", "")]:
            st.write(f"⚠️ {reason}" + (f"（{count}次）" if count != "" else ""))
    st.subheader("优先改进建议")
    for suggestion in result["suggestions"] or ["暂无需要优先改进的问题。"]:
        st.write(f"- {suggestion}")

with tab_report:
    signature = json.dumps({k: result[k] for k in ("total", "positive", "negative", "neutral", "keywords", "positive_reasons", "negative_reasons")}, ensure_ascii=False, sort_keys=True)
    if st.session_state.get("report_signature") != signature:
        try:
            st.session_state["business_report"] = generate_ai_business_report(result)
            st.session_state["report_status"] = "AI 生成"
        except Exception as exc:
            st.session_state["business_report"] = generate_business_report(result)
            st.session_state["report_status"] = f"规则版降级（AI 暂不可用：{exc}）"
        st.session_state["report_signature"] = signature
    business_report = st.session_state["business_report"]
    status = st.session_state["report_status"]
    st.subheader("AI 商家诊断报告")
    st.caption(f"报告状态：{status} · 已限制发送给模型的评论样本数量，以控制调用成本")
    report_items = list(business_report.items())
    for row_start in range(0, len(report_items), 2):
        card_cols = st.columns(2)
        for card_col, (title, content) in zip(card_cols, report_items[row_start:row_start + 2]):
            with card_col:
                st.markdown(f'<div class="report-card"><h4>{title}</h4><div>{content}</div></div>', unsafe_allow_html=True)
        st.write("")
    report_lines = ["# AI 商家诊断报告", "", f"报告状态：{status}", ""]
    for title, content in business_report.items():
        report_lines.extend([f"## {title}", "", content, ""])
    report_markdown = "\n".join(report_lines)
    report_csv = pd.DataFrame(list(business_report.items()), columns=["报告模块", "诊断内容"])
    download_cols = st.columns(3)
    with download_cols[0]:
        st.download_button("下载 Markdown 报告", report_markdown.encode("utf-8"), "商家诊断报告.md", "text/markdown", use_container_width=True)
    with download_cols[1]:
        st.download_button("下载 CSV 报告", report_csv.to_csv(index=False).encode("utf-8-sig"), "商家诊断报告.csv", "text/csv", use_container_width=True)
    with download_cols[2]:
        source_bytes = io.BytesIO()
        df.to_csv(source_bytes, index=False)
        st.download_button("下载当前数据", source_bytes.getvalue(), "评论数据.csv", "text/csv", use_container_width=True)

with tab_data:
    st.caption(f"当前读取 {len(df):,} 条评论；仅在本页面内用于分析。")
    st.dataframe(df, hide_index=True, use_container_width=True)
