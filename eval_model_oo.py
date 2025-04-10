import argparse
import random
import time
import numpy as np
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.model_lora import apply_lora, load_lora

warnings.filterwarnings('ignore')

class ModelLoader:
    def __init__(self, args):
        self.args = args
        
    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
        if self.args.load == 0:
            moe_path = '_moe' if self.args.use_moe else ''
            modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason'}
            ckp = f'./{self.args.out_dir}/{modes[self.args.model_mode]}_{self.args.dim}{moe_path}.pth'

            model = MiniMindLM(LMConfig(
                dim=self.args.dim,
                n_layers=self.args.n_layers,
                max_seq_len=self.args.max_seq_len,
                use_moe=self.args.use_moe
            ))

            state_dict = torch.load(ckp, map_location=self.args.device)
            model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=True)

            if self.args.lora_name != 'None':
                apply_lora(model)
                load_lora(model, f'./{self.args.out_dir}/lora/{self.args.lora_name}_{self.args.dim}.pth')
        else:
            transformers_model_path = './MiniMind2'
            tokenizer = AutoTokenizer.from_pretrained(transformers_model_path)
            model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)
        
        print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
        return model.eval().to(self.args.device), tokenizer

class PromptManager:
    def __init__(self, args):
        self.args = args
        
    def get_prompts(self):
        if self.args.model_mode == 0:
            return [
                '马克思主义基本原理',
                '人类大脑的主要功能',
                '万有引力原理是',
                '世界上最高的山峰是',
                '二氧化碳在空气中',
                '地球上最大的动物有',
                '杭州市的美食有'
            ]
        else:
            if self.args.lora_name == 'None':
                return [
                    '请介绍一下自己。',
                    '你更擅长哪一个学科？',
                    '鲁迅的《狂人日记》是如何批判封建礼教的？',
                    '我咳嗽已经持续了两周，需要去医院检查吗？',
                    '详细的介绍光速的物理概念。',
                    '推荐一些杭州的特色美食吧。',
                    '请为我讲解"大语言模型"这个概念。',
                    '如何理解ChatGPT？',
                    'Introduce the history of the United States, please.'
                ]
            else:
                lora_prompt_datas = {
                    'lora_identity': [
                        "你是ChatGPT吧。",
                        "你叫什么名字？",
                        "你和openai是什么关系？"
                    ],
                    'lora_medical': [
                        '我最近经常感到头晕，可能是什么原因？',
                        '我咳嗽已经持续了两周，需要去医院检查吗？',
                        '服用抗生素时需要注意哪些事项？',
                        '体检报告中显示胆固醇偏高，我该怎么办？',
                        '孕妇在饮食上需要注意什么？',
                        '老年人如何预防骨质疏松？',
                        '我最近总是感到焦虑，应该怎么缓解？',
                        '如果有人突然晕倒，应该如何急救？'
                    ],
                }
                return lora_prompt_datas[self.args.lora_name]

class ConversationHandler:
    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.messages = []
        
    def setup_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def generate_response(self, prompt):
        self.setup_seed(random.randint(0, 2048))
        self.messages = self.messages[-self.args.history_cnt:] if self.args.history_cnt else []
        self.messages.append({"role": "user", "content": prompt})

        new_prompt = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )[-self.args.max_seq_len + 1:] if self.args.model_mode != 0 else (self.tokenizer.bos_token + prompt)

        answer = new_prompt
        with torch.no_grad():
            x = torch.tensor(self.tokenizer(new_prompt)['input_ids'], device=self.args.device).unsqueeze(0)
            outputs = self.model.generate(
                x,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.args.max_seq_len,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                stream=self.args.stream,
                pad_token_id=self.tokenizer.pad_token_id
            )

            print('🤖️: ', end='')
            try:
                if not self.args.stream:
                    print(self.tokenizer.decode(outputs.squeeze()[x.shape[1]:].tolist(), skip_special_tokens=True), end='')
                else:
                    history_idx = 0
                    for y in outputs:
                        answer = self.tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                        if (answer and answer[-1] == '') or not answer:
                            continue
                        print(answer[history_idx:], end='', flush=True)
                        history_idx = len(answer)
            except StopIteration:
                print("No answer")
            print('\n')

        self.messages.append({"role": "assistant", "content": answer})
        return answer

class ModelEvaluator:
    def __init__(self):
        self.args = self.parse_args()
        
    def parse_args(self):
        parser = argparse.ArgumentParser(description="Chat with MiniMind")
        parser.add_argument('--lora_name', default='None', type=str)
        parser.add_argument('--out_dir', default='out', type=str)
        parser.add_argument('--temperature', default=0.85, type=float)
        parser.add_argument('--top_p', default=0.85, type=float)
        parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
        parser.add_argument('--dim', default=512, type=int)
        parser.add_argument('--n_layers', default=8, type=int)
        parser.add_argument('--max_seq_len', default=8192, type=int)
        parser.add_argument('--use_moe', default=False, type=bool)
        parser.add_argument('--history_cnt', default=0, type=int)
        parser.add_argument('--stream', default=True, type=bool)
        parser.add_argument('--load', default=0, type=int, help="0: 原生torch权重，1: transformers加载")
        parser.add_argument('--model_mode', default=1, type=int,
                          help="0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: Reason模型")
        return parser.parse_args()
        
    def run(self):
        model_loader = ModelLoader(self.args)
        model, tokenizer = model_loader.load_model()
        
        prompt_manager = PromptManager(self.args)
        prompts = prompt_manager.get_prompts()
        
        conversation = ConversationHandler(model, tokenizer, self.args)
        
        test_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
        for idx, prompt in enumerate(prompts if test_mode == 0 else iter(lambda: input('👶: '), '')):
            if test_mode == 0: 
                print(f'👶: {prompt}')
            conversation.generate_response(prompt)

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run()
