from typing import List

class Generator:
    def __init__(self, model, tokenizer, max_new_tokens=100, temperature=0.7, 
                 top_p=0.9, repetition_penalty=1.2):
        """初始化生成器
        Args:
            model: 语言模型
            tokenizer: 分词器
            max_new_tokens: 生成最大token数
            temperature: 生成温度
            top_p: 核采样参数
            repetition_penalty: 重复惩罚系数
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        
    def generate(self, query: str, context: List[str]) -> str:
        """基于检索内容生成回答"""
        # 构建prompt
        prompt = self._build_prompt(query, context)
        
        # 调用模型生成
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            repetition_penalty=self.repetition_penalty
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 简单幻觉检测
        if self._detect_hallucination(response, context):
            return "抱歉，我无法从提供的资料中找到确切答案。"
            
        return response
    
    def _build_prompt(self, query: str, context: List[str]) -> str:
        """构建RAG prompt模板"""
        if not context:
            return f"""问题：{query}\n"""
            
        # 提取最相关的上下文片段
        relevant_context = context[0].split('\n')[0] if context else ""
        
        return f"""根据以下信息回答问题：\n{relevant_context}\n问题：{query}\n"""
    
    def _detect_hallucination(self, response: str, context: List[str]) -> bool:
        """简单幻觉检测"""
        # 如果回答中包含"我不知道"等短语
        if any(phrase in response.lower() 
               for phrase in ["不知道", "不确定", "没有提到"]):
            return True
        
        # 这里可以扩展更复杂的检测逻辑
        return False
