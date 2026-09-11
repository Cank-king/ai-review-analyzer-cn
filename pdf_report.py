"""生成不依赖 AI 调用的中文 A4 商家诊断 PDF。"""

from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)


REPORT_FONT = "STSong-Light"


def _register_chinese_font():
    """优先使用系统/项目中的中文字体，缺少字体时回退到 ReportLab CID 字体。"""
    candidates = [
        Path(__file__).with_name("fonts") / "NotoSansSC-Regular.ttf",
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("C:/Windows/Fonts/msyh.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    ]
    for font_path in candidates:
        if font_path.suffix.lower() not in {".ttf", ".otf"} or not font_path.exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont("MerchantChinese", str(font_path)))
            return "MerchantChinese"
        except (OSError, TypeError, ValueError):
            continue
    pdfmetrics.registerFont(UnicodeCIDFont(REPORT_FONT))
    return REPORT_FONT


def _plain_content(content):
    if isinstance(content, dict):
        return "\n".join(f"{key}：{_plain_content(value)}" for key, value in content.items())
    if isinstance(content, (list, tuple, set)):
        return "\n".join(f"{index}. {_plain_content(item)}" for index, item in enumerate(content, 1))
    if content is None:
        return "暂无内容"
    return str(content).strip() or "暂无内容"


def _paragraph(text, style):
    safe = escape(str(text)).replace("\n", "<br/>")
    return Paragraph(safe, style)


def _footer(canvas, document):
    canvas.saveState()
    canvas.setFont(REPORT_FONT, 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(20 * mm, 12 * mm, "AI 电商评论经营诊断报告")
    canvas.drawRightString(190 * mm, 12 * mm, f"第 {document.page} 页")
    canvas.restoreState()


def build_pdf_report(result, business_report, report_status="规则版降级", generated_at=None):
    """使用已有分析结果和报告内容生成 PDF 字节，不触发任何 AI 调用。"""
    global REPORT_FONT
    REPORT_FONT = _register_chinese_font()
    font_name = REPORT_FONT
    generated_at = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = int(result.get("total", 0))
    positive = int(result.get("positive", 0))
    neutral = int(result.get("neutral", 0))
    negative = int(result.get("negative", 0))
    positive_rate = positive / total * 100 if total else 0
    negative_rate = negative / total * 100 if total else 0
    issue_evidence = result.get("issue_evidence", [])
    selling_evidence = result.get("selling_point_evidence", [])
    evidence_count = len(set(quote for item in issue_evidence + selling_evidence for quote in item.get("evidence", [])))
    issue_coverage = result.get("issue_coverage_count", 0) / negative * 100 if negative else None

    styles = getSampleStyleSheet()
    title = ParagraphStyle("CoverTitle", parent=styles["Title"], fontName=font_name, fontSize=25, leading=34, alignment=TA_CENTER, textColor=colors.HexColor("#17324D"), spaceAfter=18)
    subtitle = ParagraphStyle("CoverSubtitle", parent=styles["Normal"], fontName=font_name, fontSize=12, leading=20, alignment=TA_CENTER, textColor=colors.HexColor("#475569"))
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName=font_name, fontSize=18, leading=25, textColor=colors.HexColor("#17324D"), spaceBefore=10, spaceAfter=9)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName=font_name, fontSize=13, leading=20, textColor=colors.HexColor("#256B70"), spaceBefore=8, spaceAfter=5)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName=font_name, fontSize=10.5, leading=17, textColor=colors.HexColor("#263238"), spaceAfter=6)
    small = ParagraphStyle("Small", parent=body, fontSize=9, leading=14, textColor=colors.HexColor("#475569"))
    label = ParagraphStyle("Label", parent=body, fontSize=10, leading=16, textColor=colors.HexColor("#334155"), spaceAfter=3)

    story = [Spacer(1, 45 * mm), _paragraph("AI 电商评论经营诊断报告", title), _paragraph(f"报告生成日期：{generated_at[:10]}", subtitle), Spacer(1, 18 * mm)]
    story.extend([
        _paragraph(f"评论总数：{total} 条", subtitle),
        _paragraph(f"好评率：{positive_rate:.1f}%　差评率：{negative_rate:.1f}%", subtitle),
        _paragraph(f"报告状态：{report_status}", subtitle),
        PageBreak(),
        _paragraph("本次诊断依据", h1),
        _paragraph(f"分析评论数量：{total} 条", body),
        _paragraph(f"好评：{positive} 条　中评：{neutral} 条　差评：{negative} 条", body),
        _paragraph(f"AI 实际分析样本：{evidence_count} 条", body),
        _paragraph(f"主要问题覆盖率：{('样本不足' if issue_coverage is None else f'{issue_coverage:.1f}%')}", body),
        _paragraph("说明：统计数字由程序根据上传数据计算，AI 负责解释和经营建议。", small),
        Spacer(1, 5 * mm),
        _paragraph("执行摘要", h1),
        _paragraph(_plain_content(business_report.get("执行摘要", "暂无内容")), body),
    ])

    if issue_evidence:
        story.append(_paragraph("核心问题 TOP 5", h1))
        for index, item in enumerate(issue_evidence[:5], 1):
            story.append(_paragraph(f"{index}. {item.get('issue_name', '未命名问题')}", h2))
            share = "样本不足" if item.get("negative_share") is None else f"{item['negative_share']:.1f}%"
            story.append(_paragraph(f"涉及评论数：{item.get('mention_count', 0)} 条　负面评论数：{item.get('negative_count', 0)} 条　占全部差评比例：{share}", label))
            story.append(_paragraph(f"严重程度：{item.get('severity', '样本不足')}　商业影响：{item.get('commercial_impact', '暂无')}", label))
            story.append(_paragraph(f"建议动作：{item.get('recommended_action', '暂无')}", label))
            story.append(_paragraph(f"代表性真实评论：{_plain_content(item.get('evidence', []))}", small))
    if selling_evidence:
        story.append(_paragraph("消费者认可卖点 TOP 5", h1))
        for index, item in enumerate(selling_evidence[:5], 1):
            share = "样本不足" if item.get("positive_share") is None else f"{item['positive_share']:.1f}%"
            story.append(_paragraph(f"{index}. {item.get('point_name', '未命名卖点')}", h2))
            story.append(_paragraph(f"提及次数：{item.get('mention_count', 0)} 次　正面评论数量：{item.get('positive_count', 0)} 条　占全部好评比例：{share}", label))
            story.append(_paragraph(f"代表性真实评论：{_plain_content(item.get('evidence', []))}", small))

    for section in ("优先整改事项 TOP 3", "运营/营销建议", "客服/售后建议", "产品优化建议", "30 天行动清单", "商家总结"):
        story.append(_paragraph(section, h1))
        story.append(_paragraph(_plain_content(business_report.get(section, "暂无内容")), body))

    from io import BytesIO
    output = BytesIO()
    document = SimpleDocTemplate(output, pagesize=A4, rightMargin=20 * mm, leftMargin=20 * mm, topMargin=18 * mm, bottomMargin=20 * mm, title="AI 电商评论经营诊断报告", author="AI Review Analyzer")
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return output.getvalue()
