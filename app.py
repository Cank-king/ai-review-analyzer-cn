"""AI 电商评论分析器 Streamlit 应用。"""

import io
import json
import ast
import html
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from analyzer import analyze_reviews, generate_ai_business_report, generate_business_report
from pdf_report import build_pdf_report


REPORT_LABELS = ("问题", "严重程度", "建议动作", "优先原因", "为什么优先处理")


def _format_text_line(line):
    """转义文本，并将整改事项中的字段标签加粗。"""
    escaped = html.escape(str(line), quote=True)
    for label in REPORT_LABELS:
        escaped = re.sub(rf"^({re.escape(label)}：)", r"<strong>\1</strong>", escaped)
    return escaped


def format_report_content(content):
    """把模型可能返回的 list、dict、string 安全转换为可读 HTML。"""
    if isinstance(content, dict):
        items = []
        for key, value in content.items():
            items.append(f"<li><strong>{html.escape(str(key))}</strong>：{format_report_content(value)}</li>")
        return "<ul>" + "".join(items) + "</ul>" if items else "<span>暂无内容</span>"
    if isinstance(content, (list, tuple, set)):
        items = "".join(f"<li>{format_report_content(item)}</li>" for item in content)
        return "<ol>" + items + "</ol>" if items else "<span>暂无内容</span>"
    if content is None:
        return "<span>暂无内容</span>"

    text = str(content).strip()
    if text[:1] in ("[", "{") and text[-1:] in ("]", "}"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            try:
                parsed = json.loads(text)
            except (TypeError, json.JSONDecodeError):
                parsed = None
        if isinstance(parsed, (list, tuple, dict)):
            return format_report_content(parsed)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "<span>暂无内容</span>"
    list_items = []
    normal_lines = []
    for line in lines:
        match = re.match(r"^(?:[-*]|\d+[.)])\s*(.+)$", line)
        if match:
            list_items.append(_format_text_line(match.group(1)))
        else:
            normal_lines.append(_format_text_line(line))
    parts = []
    if normal_lines:
        parts.append("<br>".join(normal_lines))
    if list_items:
        parts.append("<ol>" + "".join(f"<li>{item}</li>" for item in list_items) + "</ol>")
    return "".join(parts)


def format_report_markdown(content):
    """将报告值转换为下载用 Markdown，避免出现 Python list 表示法。"""
    if isinstance(content, dict):
        return "\n".join(f"- **{key}**：{format_report_markdown(value)}" for key, value in content.items())
    if isinstance(content, (list, tuple, set)):
        return "\n".join(f"{index}. {format_report_markdown(item)}" for index, item in enumerate(content, 1))
    if content is None:
        return "暂无内容"
    text = str(content).strip()
    if text[:1] in ("[", "{") and text[-1:] in ("]", "}"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            try:
                parsed = json.loads(text)
            except (TypeError, json.JSONDecodeError):
                parsed = None
        if isinstance(parsed, (list, tuple, dict)):
            return format_report_markdown(parsed)
    return text


def _pick_column(columns, names, fallback):
    """按常见中英文列名自动选择，仍保留页面下拉选择器供用户调整。"""
    normalized = {str(column).strip().lower(): column for column in columns}
    for name in names:
        if name.lower() in normalized:
            return columns.index(normalized[name.lower()])
    return fallback


def _percent(value):
    return "样本不足" if value is None else f"{value:.1f}%"


def _evidence_card(item, is_issue=True):
    if is_issue:
        title = item["issue_name"]
        metrics = f"涉及评论：{item['mention_count']} 条 · 负面评论：{item['negative_count']} 条 · 占差评：{_percent(item['negative_share'])}"
        details = (
            f"严重程度：{item['severity']}<br>"
            f"商业影响：{html.escape(item['commercial_impact'])}<br>"
            f"建议动作：{html.escape(item['recommended_action'])}<br>"
            f"优先原因：{html.escape(item['reason'])}"
        )
    else:
        title = item["point_name"]
        metrics = f"提及次数：{item['mention_count']} 次 · 正面评论：{item['positive_count']} 条 · 占好评：{_percent(item['positive_share'])}"
        details = ""
    quotes = item.get("evidence", [])
    quote_html = format_report_content(quotes) if quotes else "<span>暂无代表性评论（样本不足）</span>"
    return (
        f'<div class="evidence-card"><h5>{html.escape(title)}</h5>'
        f'<div class="evidence-metrics">{html.escape(metrics)}</div>'
        f'<div>{details}</div><div class="evidence-label">代表性评论</div>{quote_html}</div>'
    )


st.set_page_config(page_title="商家评论诊断台", page_icon="🧭", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1200px; padding-top: 2rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#17324d,#256b70); color:white; margin-bottom:1.2rem;}
.hero h1 {margin:0 0 .35rem 0; font-size:2rem;}
.hero p {margin:0; opacity:.86; font-size:1rem;}
.report-card {padding:1.15rem 1.25rem; border:1px solid #dbe3ea; border-radius:14px; background:#fff; min-height:125px; color:#334155; line-height:1.75; margin-bottom:.5rem;}
.report-card * {color:#334155 !important;}
.report-card h4 {margin:0 0 .65rem 0; color:#17324d !important; font-size:1.05rem;}
.report-card strong {color:#263238 !important; font-weight:700;}
.report-card ul, .report-card ol {margin:.35rem 0 .2rem 1.2rem; padding-left:1rem;}
.report-card li {margin:.25rem 0;}
.evidence-basis, .evidence-card {padding:1rem 1.15rem; border:1px solid #dbe3ea; border-radius:12px; background:#f8fafc; color:#334155; line-height:1.7; margin:.6rem 0;}
.evidence-basis *, .evidence-card * {color:#334155 !important;}
.evidence-basis h4, .evidence-card h5 {margin:0 0 .35rem 0; color:#17324d !important;}
.basis-note, .evidence-metrics, .evidence-label {font-size:.9rem; color:#475569 !important;}
.evidence-label {font-weight:700; margin-top:.35rem;}
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
default_text = _pick_column(columns, ["评论内容", "评论", "评价", "内容", "review", "comment", "text"], 0)
default_rating = _pick_column(columns, ["评分", "星级", "rating", "score", "stars"], min(1, len(columns) - 1))
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
            st.dataframe(pd.DataFrame(result["keywords"], columns=["关键词", "出现次数"]), hide_index=True, width="stretch")
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
    signature = json.dumps({k: result[k] for k in ("total", "positive", "negative", "neutral", "keywords", "positive_reasons", "negative_reasons", "issue_evidence", "selling_point_evidence")}, ensure_ascii=False, sort_keys=True)
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
    issue_evidence = result.get("issue_evidence", [])
    selling_evidence = result.get("selling_point_evidence", [])
    evidence_review_count = len(set(
        quote for item in issue_evidence + selling_evidence for quote in item.get("evidence", [])
    ))
    covered_issues = len(issue_evidence)
    issue_coverage = result.get("issue_coverage_count", 0) / result["negative"] * 100 if result["negative"] else None
    st.markdown(
        f'<div class="evidence-basis"><h4>本次诊断依据</h4>'
        f'<div>分析评论：{result["total"]} 条　好评：{result["positive"]} 条　中评：{result["neutral"]} 条　差评：{result["negative"]} 条</div>'
        f'<div>AI 实际分析样本：{evidence_review_count} 条　主要问题覆盖：{_percent(issue_coverage)}</div>'
        '<div class="basis-note">AI 负责归纳、解释和经营建议；数量、比例等关键指标由程序根据上传数据计算。</div></div>',
        unsafe_allow_html=True,
    )
    if issue_evidence:
        st.markdown("#### 核心问题数据证据")
        for item in issue_evidence[:5]:
            st.markdown(_evidence_card(item), unsafe_allow_html=True)
    if selling_evidence:
        st.markdown("#### 消费者认可卖点数据证据")
        for item in selling_evidence[:5]:
            st.markdown(_evidence_card(item, is_issue=False), unsafe_allow_html=True)
    report_items = list(business_report.items())
    for row_start in range(0, len(report_items), 2):
        card_cols = st.columns(2)
        for card_col, (title, content) in zip(card_cols, report_items[row_start:row_start + 2]):
            with card_col:
                safe_content = format_report_content(content)
                st.markdown(f'<div class="report-card"><h4>{html.escape(str(title))}</h4><div>{safe_content}</div></div>', unsafe_allow_html=True)
        st.write("")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        "# AI 商家诊断报告", "", f"报告状态：{status}", f"生成时间：{generated_at}",
        f"评论总数：{result['total']}", f"好评率：{positive_rate:.1f}%", f"差评率：{negative_rate:.1f}%",
        f"中评数：{result['neutral']}", f"AI 实际分析样本：{evidence_review_count}", f"主要问题覆盖：{_percent(issue_coverage)}", "",
    ]
    if issue_evidence:
        report_lines.extend(["## 核心问题数据证据", ""])
        for item in issue_evidence[:5]:
            report_lines.extend([
                f"### {item['issue_name']}",
                f"涉及评论：{item['mention_count']} 条；负面评论：{item['negative_count']} 条；占差评：{_percent(item['negative_share'])}",
                f"严重程度：{item['severity']}", f"商业影响：{item['commercial_impact']}",
                f"建议动作：{item['recommended_action']}", f"优先原因：{item['reason']}",
                "代表性评论：", format_report_markdown(item.get("evidence", [])), "",
            ])
    if selling_evidence:
        report_lines.extend(["## 消费者认可卖点数据证据", ""])
        for item in selling_evidence[:5]:
            report_lines.extend([
                f"### {item['point_name']}",
                f"提及次数：{item['mention_count']} 次；正面评论：{item['positive_count']} 条；占好评：{_percent(item['positive_share'])}",
                "代表性评论：", format_report_markdown(item.get("evidence", [])), "",
            ])
    for title, content in business_report.items():
        report_lines.extend([f"## {title}", "", format_report_markdown(content), ""])
    report_markdown = "\n".join(report_lines)
    report_csv = pd.DataFrame(
        [(title, format_report_markdown(content)) for title, content in business_report.items()],
        columns=["报告模块", "诊断内容"],
    )
    classified_csv = result["data"].to_csv(index=False).encode("utf-8-sig")
    pdf_bytes = build_pdf_report(result, business_report, status, generated_at)
    download_cols = st.columns(4)
    with download_cols[0]:
        st.download_button("下载完整诊断报告（Markdown）", report_markdown.encode("utf-8"), "merchant_review_report.md", "text/markdown", width="stretch")
    with download_cols[1]:
        st.download_button("下载诊断摘要（CSV）", report_csv.to_csv(index=False).encode("utf-8-sig"), "merchant_review_report.csv", "text/csv", width="stretch")
    with download_cols[2]:
        source_bytes = io.BytesIO()
        df.to_csv(source_bytes, index=False)
        st.download_button("下载原始评论（含分类）", classified_csv, "merchant_review_classified_reviews.csv", "text/csv", width="stretch")
    with download_cols[3]:
        st.download_button(
            "📄 下载 PDF 诊断报告",
            pdf_bytes,
            f"merchant_review_diagnosis_{datetime.now():%Y%m%d}.pdf",
            "application/pdf",
            width="stretch",
        )

with tab_data:
    st.caption(f"当前读取 {len(df):,} 条评论；仅在本页面内用于分析。")
    st.dataframe(df, hide_index=True, width="stretch")
