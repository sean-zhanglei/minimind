import faiss
import numpy as np
import os
import pickle
from typing import List

class VectorDB:
    def __init__(self, dimension: int = 768):
        """初始化FAISS向量数据库"""
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.text_chunks = []
        self.cache_dir = "rag/vector_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def store_embeddings(self, embeddings: np.ndarray, chunks: List[str]):
        """存储向量和对应的文本块"""
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks must have same length")
        
        # 转换为float32数组
        embeddings = embeddings.astype('float32')
        
        # 直接添加到索引
        self.index.add(embeddings)
            
        self.text_chunks.extend(chunks)

    def query(self, query_embedding: np.ndarray, k: int = 3) -> List[str]:
        """查询最相似的k个文本块"""
        if len(self.text_chunks) == 0:
            return []
            
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        
        # 过滤无效索引
        valid_indices = [i for i in indices[0] if 0 <= i < len(self.text_chunks)]
        return [self.text_chunks[i] for i in valid_indices]
        
    def save_to_disk(self, file_hash: str):
        """保存向量数据库到磁盘"""
        cache_file = os.path.join(self.cache_dir, f"{file_hash}.pkl")
        
        # 序列化数据
        data = {
            'dimension': self.dimension,
            'index': faiss.serialize_index(self.index),
            'text_chunks': self.text_chunks
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
            
    def load_from_disk(self, file_hash: str) -> bool:
        """从磁盘加载向量数据库"""
        cache_file = os.path.join(self.cache_dir, f"{file_hash}.pkl")
        
        if not os.path.exists(cache_file):
            return False
            
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            
        self.dimension = data['dimension']
        self.index = faiss.deserialize_index(data['index'])
        self.text_chunks = data['text_chunks']
        return True
        
    def get_file_hash(self, file_path: str) -> str:
        """生成文件唯一哈希"""
        import hashlib
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
