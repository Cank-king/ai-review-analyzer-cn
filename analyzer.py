"""AI 电商评论分析器的核心分析函数。"""

import re
from collections import Counter

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
