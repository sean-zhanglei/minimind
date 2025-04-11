import os
import torch
import json
from typing import Optional
from dataclasses import dataclass

@dataclass
class RAGConfig:
    """RAG系统配置类"""
    # 模型配置
    model_path: str = "MiniMind2"
    tokenizer_path: str = "MiniMind2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 检索配置
    chunk_size: int = 1024  # 文本分块大小
    chunk_overlap: int = 50  # 分块重叠大小
    top_k: int = 3  # 检索返回的上下文数量
    
    # 生成配置
    max_new_tokens: int = 100  # 生成最大token数
    temperature: float = 0.7  # 生成温度
    top_p: float = 0.9  # 核采样参数
    repetition_penalty: float = 1.2  # 重复惩罚
    
    # 调试配置
    debug: bool = False  # 调试模式
    log_level: str = "INFO"  # 日志级别
    
    # PDF处理配置
    pdf_extract_mode: str = "text"  # text或layout
    
    @classmethod
    def from_json(cls, json_path: Optional[str] = None):
        """从JSON文件加载配置"""
        if json_path and os.path.exists(json_path):
            with open(json_path, "r") as f:
                config_data = json.load(f)
            return cls(**config_data)
        return cls()

# 默认配置实例
default_config = RAGConfig()
