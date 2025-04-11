from .retriever import Retriever
from .generator import Generator
from .config import RAGConfig
from MiniMind2.model import MiniMindLM
from MiniMind2.LMConfig import LMConfig
from transformers import AutoTokenizer
import torch
import argparse
import json

class RAGSystem:
    def __init__(self, config: RAGConfig = None):
        """初始化RAG系统
        Args:
            config: RAG配置对象，如果为None则使用默认配置
        """
        self.config = config if config else RAGConfig()
        
        try:
            # 加载模型配置
            with open("MiniMind2/config.json", "r") as f:
                config_data = json.load(f)
            lm_config = LMConfig(**config_data)
            
            # 初始化模型和分词器
            self.model = MiniMindLM(lm_config)
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
            
            # 确保模型在评估模式
            self.model.eval()
            
        except Exception as e:
            print(f"初始化失败: {str(e)}")
            raise
        self.retriever = Retriever(self.model, self.tokenizer, 
                                  chunk_size=self.config.chunk_size,
                                  chunk_overlap=self.config.chunk_overlap,
                                  top_k=self.config.top_k)
        self.generator = Generator(self.model, self.tokenizer,
                                 max_new_tokens=self.config.max_new_tokens,
                                 temperature=self.config.temperature,
                                 top_p=self.config.top_p,
                                 repetition_penalty=self.config.repetition_penalty)
    
    def process_document(self, file_path: str, batch_size: int = 100):
        """处理PDF文档
        Args:
            file_path: PDF文件路径
            batch_size: 分批处理的大小
        """
        self.retriever.process_pdf(file_path, batch_size)
        print(f"成功处理文档: {file_path} (批大小: {batch_size})")
    
    def query(self, question: str) -> str:
        """回答用户问题"""
        # 检索相关上下文
        context = self.retriever.retrieve(question)
        
        # 生成回答
        return self.generator.generate(question, context)

def main():
    parser = argparse.ArgumentParser(description="RAG系统")
    parser.add_argument("--file", type=str, default="rag/test.pdf", help="PDF文件路径")
    parser.add_argument("--query", type=str, default="线程的实现方式", help="查询问题")
    parser.add_argument("--config", type=str, default="rag/example_config.json", help="配置文件路径")
    parser.add_argument("--batch-size", type=int, default=100, help="处理PDF的批大小")
    args = parser.parse_args()
    
    # 加载配置
    config = RAGConfig.from_json(args.config) if args.config else None
    print(f"加载配置: {args.config}", config)
    rag = RAGSystem(config)
    rag.process_document(args.file, args.batch_size)
    
    answer = rag.query(args.query)
    print("回答:", answer)

if __name__ == "__main__":
    main()
