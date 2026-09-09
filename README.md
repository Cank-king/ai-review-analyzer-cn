# AI 电商评论分析器

一个基于 Streamlit、pandas 和 jieba 的简体中文电商评论分析工具。

## 启动

在项目目录执行：

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

也可以直接使用：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

应用支持上传 CSV，也可以直接使用项目中的 `sample_reviews.csv`。CSV 至少需要包含评论内容列和评分列，页面可自行选择对应列。
