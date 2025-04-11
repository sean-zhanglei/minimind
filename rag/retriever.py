from .pdf_processor import PDFProcessor
from .vector_db import VectorDB
import numpy as np
import torch
from typing import List

class Retriever:
    def __init__(self, model, tokenizer, chunk_size=1024, chunk_overlap=50, top_k=3):
        """初始化检索器
        Args:
            model: 语言模型
            tokenizer: 分词器
            chunk_size: 文本分块大小
            chunk_overlap: 分块重叠大小
            top_k: 检索返回的上下文数量
        """
        self.chunk_size = chunk_size
        self.pdf_processor = PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_db = VectorDB()
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = top_k

    def process_pdf(self, file_path: str, batch_size: int = 100):
        """处理PDF文件并存储到向量数据库
        Args:
            file_path: PDF文件路径
            batch_size: 分批处理的大小
        """
        print(f"开始处理PDF文件: {file_path}")
        
        # 检查是否有缓存
        file_hash = self.vector_db.get_file_hash(file_path)
        if self.vector_db.load_from_disk(file_hash):
            print("从缓存加载向量数据库")
            return
        
        # 提取文本
        text = self.pdf_processor.extract_text(file_path)
        print("PDF文本提取完成")
        
        # 分块处理
        chunks = self.pdf_processor.chunk_text(text)
        print(f"共分割为 {len(chunks)} 个文本块")
        
        # 分批处理
        processed = 0
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            processed += len(batch_chunks)
            print(f"处理进度: {processed}/{len(chunks)} 块 ({processed/len(chunks):.1%})")
            
            # 生成嵌入
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_chunks, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512  # 使用固定长度避免内存问题
                ).to(self.device)
                
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state.mean(dim=1)
                embeddings = last_hidden.cpu().numpy()
            
            # 存储到向量数据库
            self.vector_db.store_embeddings(embeddings, batch_chunks)
            
            # 每批处理完保存到磁盘
            self.vector_db.save_to_disk(file_hash)
            print(f"已保存第 {processed//batch_size + 1} 批数据到缓存")

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """检索相关文本块"""
        # 生成查询嵌入
        with torch.no_grad():
            inputs = self.tokenizer(
                [query], 
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs)
            query_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        # 查询向量数据库
        return self.vector_db.query(query_embedding, k)
