# RAG 系统

基于检索增强生成的PDF问答系统，使用MiniMind2模型和FAISS向量数据库。

## 功能特性

- PDF文档内容提取和分块处理
- 文本向量化存储和检索
- 检索增强生成回答
- 简单幻觉检测机制

## 使用方法

1. 安装依赖：
```bash
pip install PyPDF2 faiss-cpu
```

2. 处理PDF文档：
```bash
python -m rag.main --file 文档.pdf
```

3. 查询问题：
```bash
python -m rag.main --query "你的问题"
```

4. 完整流程示例：
```bash
# 处理文档
python -m rag.main --file example.pdf

# 查询问题
python -m rag.main --query "文档中提到了哪些关键技术?"
```

## 配置选项

- 修改`pdf_processor.py`中的`chunk_size`调整文本分块大小
- 修改`retriever.py`中的`k`值调整检索结果数量
- 自定义`generator.py`中的prompt模板优化生成效果
