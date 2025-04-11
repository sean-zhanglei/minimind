import PyPDF2
from typing import List

class PDFProcessor:
    def __init__(self, chunk_size=512, chunk_overlap=50):
        """初始化PDF处理器
        Args:
            chunk_size: 文本分块大小
            chunk_overlap: 分块重叠大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text(self, file_path: str) -> str:
        """从PDF文件中提取文本内容"""
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text

    def chunk_text(self, text: str) -> List[str]:
        """将文本分块处理
        使用滑动窗口方法实现重叠分块
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        overlap_words = []
        
        for word in words:
            if current_length + len(word) + 1 > self.chunk_size:
                chunks.append(' '.join(current_chunk))
                
                # 处理重叠部分
                overlap_words = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else []
                current_chunk = overlap_words.copy()
                current_length = len(' '.join(current_chunk)) + 1 if current_chunk else 0
                
            current_chunk.append(word)
            current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
