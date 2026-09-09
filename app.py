"""AI 电商评论分析器 Streamlit 应用。"""

from pathlib import Path

import pandas as pd
import streamlit as st

from analyzer import analyze_reviews


st.set_page_config(page_title="AI 电商评论分析器", page_icon="📊", layout="wide")
st.title("AI 电商评论分析器")
st.caption("上传评论 CSV，快速了解商品口碑与改进方向。")

uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("已加载上传文件。")
    except Exception as exc:
        st.error(f"CSV 读取失败：{exc}")
        st.stop()
else:
    sample_path = Path(__file__).with_name("sample_reviews.csv")
    df = pd.read_csv(sample_path)
    st.info("当前使用示例数据演示；上传 CSV 可分析自己的评论。")

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

metrics = st.columns(4)
metrics[0].metric("评论总数", result["total"])
metrics[1].metric("好评数", result["positive"])
metrics[2].metric("差评数", result["negative"])
metrics[3].metric("中性评论数", result["neutral"])

st.subheader("中文高频关键词")
if result["keywords"]:
    st.dataframe(pd.DataFrame(result["keywords"], columns=["关键词", "出现次数"]), hide_index=True, use_container_width=True)
else:
    st.write("暂未提取到有效关键词。")

col_good, col_bad = st.columns(2)
with col_good:
    st.subheader("常见好评原因")
    if result["positive_reasons"]:
        for reason, count in result["positive_reasons"]:
            st.write(f"- {reason}（{count}次）")
    else:
        st.write("暂无明显好评原因。")
with col_bad:
    st.subheader("常见差评原因")
    if result["negative_reasons"]:
        for reason, count in result["negative_reasons"]:
            st.write(f"- {reason}（{count}次）")
    else:
        st.write("暂无明显差评原因。")

st.subheader("商品改进建议")
if result["suggestions"]:
    for suggestion in result["suggestions"]:
        st.write(f"- {suggestion}")
else:
    st.write("暂无需要优先改进的问题。")

with st.expander("查看已读取的评论数据"):
    st.dataframe(df, hide_index=True, use_container_width=True)
