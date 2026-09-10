"""AI 电商评论分析器的核心分析函数。"""

import re
import json
import os
from pathlib import Path
from collections import Counter
from urllib import error, request

import jieba
import pandas as pd


STOPWORDS = {
    "这个", "那个", "东西", "商品", "感觉", "真的", "非常", "比较", "还是", "已经",
    "可以", "一个", "没有", "不是", "就是", "而且", "但是", "因为", "所以", "买了",
    "收到", "使用", "用了", "觉得", "有点", "很", "也", "都", "还", "挺", "的", "了",
}

POSITIVE_RULES = {
    "质量好": ["质量", "做工", "材质", "结实", "耐用", "扎实"],
    "价格实惠": ["便宜", "实惠", "划算", "性价比", "优惠", "值得"],
    "物流快速": ["物流", "快递", "发货", "送货", "配送"],
    "使用方便": ["方便", "好用", "简单", "易用", "顺手", "好操作"],
    "外观满意": ["好看", "漂亮", "外观", "颜值", "颜色", "设计"],
    "服务周到": ["客服", "服务", "态度", "耐心", "回复"],
}

NEGATIVE_RULES = {
    "质量问题": ["质量", "损坏", "破损", "坏了", "故障", "瑕疵", "漏水", "开线"],
    "物流问题": ["物流", "快递", "发货", "配送", "慢", "延迟", "包装"],
    "尺寸不合适": ["尺寸", "大小", "偏大", "偏小", "不合身", "尺码"],
    "使用体验不佳": ["难用", "不好用", "麻烦", "噪音", "卡顿", "异味", "耗电"],
    "描述不符": ["描述", "不符", "不一样", "色差", "图片", "虚假"],
    "售后服务问题": ["客服", "售后", "退货", "退款", "处理", "回复"],
    "价格不满意": ["贵", "价格", "不值", "性价比低"],
}


def _as_text(value):
    return "" if pd.isna(value) else str(value)


def normalize_rating(value):
    """将评分转换为 1～5 的数字，无法识别时返回 None。"""
    try:
        rating = float(str(value).strip().replace("星", ""))
        return rating if 1 <= rating <= 5 else None
    except (TypeError, ValueError):
        return None


def classify_review(rating):
    if rating is None:
        return "中性评论"
    if rating >= 4:
        return "好评"
    if rating <= 2:
        return "差评"
    return "中性评论"


def analyze_reviews(df, text_column, rating_column, top_n=15):
    """分析评论数据，返回统计、关键词、原因和建议。"""
    if text_column not in df.columns or rating_column not in df.columns:
        raise ValueError("评论内容列和评分列不能为空")

    data = df[[text_column, rating_column]].copy()
    data[text_column] = data[text_column].map(_as_text)
    data["_评分"] = data[rating_column].map(normalize_rating)
    data["_分类"] = data["_评分"].map(classify_review)
    texts = data[text_column].tolist()
    words = []
    for text in texts:
        words.extend(
            word for word in jieba.cut(text)
            if len(word.strip()) >= 2
            and word.strip() not in STOPWORDS
            and not re.fullmatch(r"[\W_]+", word.strip(), re.UNICODE)
            and not word.strip().isdigit()
        )
    keywords = Counter(words).most_common(top_n)
    positive_text = " ".join(data.loc[data["_分类"] == "好评", text_column])
    negative_text = " ".join(data.loc[data["_分类"] == "差评", text_column])
    positive_reasons = find_reasons(positive_text, POSITIVE_RULES)
    negative_reasons = find_reasons(negative_text, NEGATIVE_RULES)
    return {
        "data": data,
        "total": len(data),
        "positive": int((data["_分类"] == "好评").sum()),
        "negative": int((data["_分类"] == "差评").sum()),
        "neutral": int((data["_分类"] == "中性评论").sum()),
        "keywords": keywords,
        "positive_reasons": positive_reasons,
        "negative_reasons": negative_reasons,
        "suggestions": generate_suggestions(negative_reasons),
    }


def find_reasons(text, rules):
    return [(reason, sum(text.count(word) for word in words)) for reason, words in rules.items()
            if any(word in text for word in words)]


def generate_suggestions(negative_reasons):
    suggestions = {
        "质量问题": "加强出厂质检，重点检查破损、漏水和开线等问题。",
        "物流问题": "优化仓储和配送流程，并加强运输包装与物流时效管理。",
        "尺寸不合适": "完善尺寸说明和测量指引，提供更清晰的尺码对照表。",
        "使用体验不佳": "收集具体使用场景，优化产品功能、噪音和操作体验。",
        "描述不符": "核对商品详情页与实物，补充真实图片、参数和色差说明。",
        "售后服务问题": "设置明确的售后响应时限，提升客服培训和问题跟进。",
        "价格不满意": "重新评估定价与促销策略，突出产品价值并提供合理优惠。",
    }
    return [suggestions[reason] for reason, _ in sorted(negative_reasons, key=lambda x: x[1], reverse=True)]


def generate_business_report(result):
    """根据评论分析结果生成规则版正式商家诊断报告。"""
    positive_reasons = sorted(result.get("positive_reasons", []), key=lambda item: item[1], reverse=True)
    negative_reasons = sorted(result.get("negative_reasons", []), key=lambda item: item[1], reverse=True)
    suggestions = result.get("suggestions", [])
    keywords = result.get("keywords", [])
    total = int(result.get("total", 0))
    positive = int(result.get("positive", 0))
    negative = int(result.get("negative", 0))

    focus = "、".join(reason for reason, _ in negative_reasons[:5]) or "暂无集中出现的问题"
    selling_points = "、".join(reason for reason, _ in positive_reasons[:5])
    if not selling_points:
        selling_points = "、".join(word for word, _ in keywords[:5]) or "暂无明确卖点"

    issue_lines = []
    for index, (reason, count) in enumerate(negative_reasons[:5]):
        severity = "高" if count >= 3 or (negative and count / max(negative, 1) >= 0.5) else "中"
        action = suggestions[index] if index < len(suggestions) else "建立专项跟进和复盘机制"
        issue_lines.append(f"- {reason}（{count}次，严重程度：{severity}）：建议动作：{action}；优先原因：该问题在差评中重复出现，直接影响购买信心。")
    core_issues = "\n".join(issue_lines) or "- 暂无可识别的核心问题。"

    improvement_lines = []
    for index, (reason, count) in enumerate(negative_reasons[:3]):
        severity = "高" if count >= 3 else "中"
        action = suggestions[index] if index < len(suggestions) else "安排负责人在本周内制定改进方案"
        improvement_lines.append(f"- 问题：{reason}；严重程度：{severity}；建议动作：{action}；为什么优先处理：出现频次较高，需要先降低差评来源。")
    priority = "\n".join(improvement_lines) or "- 暂无优先整改事项，持续观察评论变化。"

    if total:
        positive_rate = positive / total * 100
        negative_rate = negative / total * 100
        summary = f"共分析 {total} 条评论，好评率 {positive_rate:.1f}%，差评率 {negative_rate:.1f}%。建议先解决高频问题，再放大稳定的正面体验。"
    else:
        positive_rate = negative_rate = 0
        summary = "当前没有可供分析的评论，暂时无法形成商家总结。"

    return {
        "执行摘要": f"本次分析 {total} 条评论，消费者主要关注：{focus}；商家可重点宣传：{selling_points}。",
        "核心问题 TOP 5": core_issues,
        "消费者认可卖点 TOP 5": "、".join(reason for reason, _ in positive_reasons[:5]) or "暂无明确认可卖点。",
        "优先整改事项 TOP 3": priority,
        "运营/营销建议": f"围绕“{selling_points}”制作真实场景内容，展示具体体验和用户证据；对“{focus}”相关问题在详情页主动说明改进措施。",
        "客服/售后建议": "建立差评问题标签和响应时限；客服回复时先确认问题，再给出明确处理节点，并对高频问题定期复盘。",
        "产品优化建议": "优先围绕高频差评原因改进产品和包装，完成小批量验证后再扩大调整范围。" if negative_reasons else "继续收集具体使用场景，验证产品体验后再安排优化。",
        "30 天行动清单": "第 1 周：确认 TOP 3 问题和负责人；第 2 周：完成客服话术、详情页和流程调整；第 3 周：验证产品或包装改进；第 4 周：复盘新评论与差评率变化。",
        "商家总结": summary,
    }


REPORT_SECTIONS = (
    "执行摘要", "核心问题 TOP 5", "消费者认可卖点 TOP 5", "优先整改事项 TOP 3",
    "运营/营销建议", "客服/售后建议", "产品优化建议", "30 天行动清单", "商家总结",
)


def _load_dashscope_api_key():
    """优先读取环境变量；若不存在，再读取项目根目录 .env。"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        return api_key.strip()
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return ""
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == "DASHSCOPE_API_KEY":
            return value.strip().strip('"').strip("'")
    return ""


def _report_input(result, max_reviews=80):
    """整理发送给模型的统计和评论，限制评论数量以控制成本。"""
    data = result.get("data")
    reviews = []
    if data is not None:
        text_column = next((column for column in data.columns if column not in {"_评分", "_分类"}), None)
        if text_column:
            reviews = [str(text) for text in data[text_column].dropna().tolist() if str(text).strip()][:max_reviews]
    return {
        "统计": {
            "评论总数": result.get("total", 0),
            "好评数": result.get("positive", 0),
            "差评数": result.get("negative", 0),
            "中性评论数": result.get("neutral", 0),
            "高频关键词": result.get("keywords", []),
            "好评原因": result.get("positive_reasons", []),
            "差评原因": result.get("negative_reasons", []),
        },
        "评论样本": reviews,
    }


def _parse_report_response(content):
    """解析模型返回的 JSON，并确保四个报告部分都存在。"""
    content = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
    if fenced:
        content = fenced.group(1).strip()
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("模型返回格式不是对象")
    report = {section: str(parsed.get(section, "")).strip() for section in REPORT_SECTIONS}
    if any(not value for value in report.values()):
        raise ValueError("模型返回缺少报告内容")
    return report


def generate_ai_business_report(result, model="qwen-turbo", timeout=30):
    """调用 DashScope OpenAI 兼容接口生成商家诊断报告。

    失败时抛出异常，由页面层自动降级到 generate_business_report()。
    """
    api_key = _load_dashscope_api_key()
    if not api_key:
        raise RuntimeError("未找到 DASHSCOPE_API_KEY")

    system_prompt = (
        "你是中文电商运营分析师。请根据评论统计和评论样本，生成正式、简洁、具体、可执行的商家诊断报告。"
        "只返回一个合法 JSON 对象，不要 Markdown，不要额外说明。对象必须包含四个键："
        "执行摘要、核心问题 TOP 5、消费者认可卖点 TOP 5、优先整改事项 TOP 3、运营/营销建议、客服/售后建议、产品优化建议、30 天行动清单、商家总结。"
        "每个值使用简体中文；核心问题和卖点最多五条，优先整改最多三条。每个优先整改事项必须包含问题、严重程度、建议动作、为什么优先处理。避免编造统计中没有的信息。"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(_report_input(result), ensure_ascii=False)},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    api_request = request.Request(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(api_request, timeout=timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"DashScope API 调用失败：{exc}") from exc

    try:
        content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("DashScope API 返回内容不完整") from exc
    return _parse_report_response(content)
