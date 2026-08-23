#!/usr/bin/env python3

import os

# 避免 NCCL "unhandled cuda error"：禁用 P2P 与 NVLS
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_NVLS_ENABLE"] = "0"

import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import logging
from collections import defaultdict

# 兼容 vLLM 0.8.x 与新版 transformers：新版 transformers 移除了 all_special_tokens_extended，
# 但 vLLM 的 get_cached_tokenizer 仍会访问该属性（如 Qwen2Tokenizer）。在导入 vLLM 前为基类补上该属性。
def _patch_transformers_for_vllm():
    import transformers.tokenization_utils_base as _tokenizer_base
    if not hasattr(_tokenizer_base.PreTrainedTokenizerBase, "all_special_tokens_extended"):
        @property
        def all_special_tokens_extended(self):
            return self.all_special_tokens
        _tokenizer_base.PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended


_patch_transformers_for_vllm()

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MATH_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def format_prompt(question: str, system_prompt: str = None) -> str:
    if system_prompt is None:
        system_prompt = DEFAULT_MATH_PROMPT
    return f"{system_prompt}\n\n{question}"

def extract_thinking(text: str) -> Tuple[str, str]:
    end_tag = "</think>"
    
    if end_tag in text:
        parts = text.split(end_tag, 1)
        
        reasoning_content = parts[0].strip()
        reasoning_content = reasoning_content.replace("<think>", "").strip()
        
        final_text = parts[1].strip() if len(parts) > 1 else ""
        
        return final_text, reasoning_content
    else:
        return text.strip(), ""

def load_existing_output(output_file: str) -> Dict[str, Any]:
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        valid_completions = []
        total_completions = data.get('completions', [])
        for completion in total_completions:
            text = completion.get('text', '')
            reasoning = completion.get('reasoning_content', '')
            if (text.strip() or reasoning.strip()) and "API call failed" not in text:
                valid_completions.append(completion)
        return {
            'data': data,
            'total_completions': len(total_completions),
            'valid_completions': len(valid_completions),
            'valid_completion_list': valid_completions
        }
    except Exception:
        return {
            'data': None,
            'total_completions': 0,
            'valid_completions': 0,
            'valid_completion_list': []
        }

def save_completed_questions(
    questions_map: Dict[int, Dict[str, Any]], 
    completion_results: Dict[int, List[Dict[str, Any]]],
    output_dir: str
) -> List[int]:
    output_path = Path(output_dir)
    saved_questions = []
    
    for original_idx, new_completions in completion_results.items():
        if not new_completions:
            continue
            
        item = questions_map.get(original_idx)
        if not item:
            continue

        question_id = item.get('question_id', item.get('id', f'q_{original_idx}'))
        output_file = output_path / f"{original_idx}.json"
        
        existing_valid_completions = []
        if output_file.exists():
            existing_data = load_existing_output(str(output_file))
            existing_valid_completions = existing_data['valid_completion_list']
        
        all_completions = existing_valid_completions + new_completions
        
        result = {
            'index': original_idx,
            'question_id': question_id,
            'question': item.get('question', ''),
            'answer': item.get('answer', ''),
            'completions': all_completions,
            'n_completions': len(all_completions),
        }
        
        for key, value in item.items():
            if key not in ['question_id', 'id', 'question', 'answer']:
                result[key] = value
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        saved_questions.append(original_idx)
        logger.info(f"Saved question {original_idx}: Total {len(all_completions)} (New: {len(new_completions)})")
    
    return saved_questions

def analyze_completion_status(
    data: List[Dict[str, Any]],
    output_dir: str,
    n_completions: int,
    start_idx: int = 0
) -> Tuple[List[Tuple[int, Dict[str, Any]]], Dict[int, int]]:
    output_path = Path(output_dir)
    pending_questions = []
    completion_needed = {}
    
    for idx, item in enumerate(data):
        original_idx = start_idx + idx
        output_file = output_path / f"{original_idx}.json"
        
        needed = n_completions
        if output_file.exists():
            result = load_existing_output(str(output_file))
            valid_count = result['valid_completions']
            if valid_count >= n_completions:
                continue 
            needed = n_completions - valid_count
            logger.info(f"Question {original_idx} needs: {needed} (Existing: {valid_count})")
        
        completion_needed[original_idx] = needed
        pending_questions.append((original_idx, item))
    
    return pending_questions, completion_needed

def _hf_dtype(dtype: str):
    import torch
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return "auto"

def batch_inference_hf(
    model_name: str,
    input_file: str,
    output_dir: str,
    n_completions: int = 64,
    batch_size: int = 1,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 2048,
    system_prompt: str = None,
    start_idx: int = 0,
    end_idx: int = None,
    device_map: str = "auto",
    dtype: str = "auto",
):
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise ImportError("HF backend selected but transformers is not installed: pip install transformers")

    import torch

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading data: {input_file}")
    data = load_jsonl(input_file)

    if end_idx is None:
        end_idx = len(data)

    current_data_slice = data[start_idx:end_idx]

    print("Analyzing existing completions status...")
    pending_questions, completion_needed = analyze_completion_status(
        current_data_slice, output_dir, n_completions, start_idx
    )

    if not pending_questions:
        print("All questions completed!")
        return

    questions_map = {idx: item for idx, item in pending_questions}

    print(f"Loading HF model: {model_name} (device_map={device_map}, dtype={dtype})")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=_hf_dtype(dtype),
    )
    model.eval()

    total_needed = sum(completion_needed.get(q_idx, 0) for q_idx, _ in pending_questions)
    pbar = tqdm(total=total_needed, desc="HF sampling", unit="gen")

    batch_results = defaultdict(list)

    # HF generation is slow; generate sequentially (batch_size is kept for interface only).
    for original_idx, item in pending_questions:
        question = item.get("question", "")
        messages = [
            {"role": "system", "content": DEFAULT_MATH_PROMPT if system_prompt is None else system_prompt},
            {"role": "user", "content": question},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt")
        inp = {k: v.to(model.device) for k, v in inp.items()}

        needed = completion_needed.get(original_idx, 0)
        if needed <= 0:
            continue

        for _ in range(needed):
            with torch.no_grad():
                out = model.generate(
                    **inp,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=tok.eos_token_id,
                )
            gen_ids = out[0][inp["input_ids"].shape[1] :]
            gen_text = tok.decode(gen_ids, skip_special_tokens=True)

            final_text, reasoning_content = extract_thinking(gen_text)
            batch_results[original_idx].append(
                {
                    "text": final_text,
                    "reasoning_content": reasoning_content,
                    "tokens": int(gen_ids.numel()),
                    "finish_reason": "eos" if (tok.eos_token_id in gen_ids.tolist()) else "length",
                }
            )
            pbar.update(1)

        # save per question to support resume
        save_completed_questions(questions_map, {original_idx: batch_results[original_idx]}, output_dir)
        batch_results.pop(original_idx, None)

    pbar.close()
    print(f"\nHF inference completed! Results saved to: {output_dir}")

def batch_inference(
    model_name: str,
    input_file: str,
    output_dir: str,
    n_completions: int = 64,
    batch_size: int = 8,
    tensor_parallel_size: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 2048,
    system_prompt: str = None,
    start_idx: int = 0,
    end_idx: int = None,
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.85,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data: {input_file}")
    data = load_jsonl(input_file)
    
    if end_idx is None:
        end_idx = len(data)
    
    current_data_slice = data[start_idx:end_idx]
    
    print(f"Analyzing existing completions status...")
    pending_questions, completion_needed = analyze_completion_status(
        current_data_slice, output_dir, n_completions, start_idx
    )
    
    if not pending_questions:
        print("All questions completed!")
        return

    questions_map = {idx: item for idx, item in pending_questions}
    
    print(f"Initializing vLLM engine (TP={tensor_parallel_size})...")
    if LLM is None or SamplingParams is None:
        raise ImportError("vLLM backend selected but vllm is not installed: pip install vllm")
    llm_kwargs = dict(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=False,
    )
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )

    print(f"Starting inference: {len(pending_questions)} questions pending")

    total_chunks = (len(pending_questions) + batch_size - 1) // batch_size
    
    for i in range(total_chunks):
        chunk_start = i * batch_size
        chunk_end = min(chunk_start + batch_size, len(pending_questions))
        current_chunk = pending_questions[chunk_start:chunk_end]
        
        print(f"Processing batch {i+1}/{total_chunks} (Questions: {len(current_chunk)})...")
        
        prompts = []
        prompt_metadata = []
        
        for original_idx, item in current_chunk:
            question = item.get('question', '')
            messages = [
                {"role": "system", "content": DEFAULT_MATH_PROMPT if system_prompt is None else system_prompt},
                {"role": "user", "content": question}
            ]
            
            full_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            needed = completion_needed.get(original_idx, 0)
            for _ in range(needed):
                prompts.append(full_prompt)
                prompt_metadata.append(original_idx)
        
        if not prompts:
            continue

        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        
        batch_results = defaultdict(list)
        
        for output, q_idx in zip(outputs, prompt_metadata):
            generated_text = output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason
            
            final_text, reasoning_content = extract_thinking(generated_text)
            
            result = {
                'text': final_text,
                'reasoning_content': reasoning_content,
                'tokens': len(output.outputs[0].token_ids),
                'finish_reason': finish_reason
            }
            batch_results[q_idx].append(result)
        
        save_completed_questions(questions_map, batch_results, output_dir)
        
        del outputs
        del batch_results

    print(f"\nInference completed! Results saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Offline Batch Inference (vLLM or HF)')
    parser.add_argument('--model', '-m', type=str, required=True, help='Model path')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output directory')
    parser.add_argument('--n-completions', '-n', type=int, default=64)
    parser.add_argument('--batch-size', '-b', type=int, default=8, help='Save every N questions')
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=8, help='vLLM: number of GPUs')
    parser.add_argument('--temperature', '-t', type=float, default=0.7)
    parser.add_argument('--top-p', '-p', type=float, default=0.95)
    parser.add_argument('--top-k', '-k', type=int, default=20)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--system-prompt', type=str, default=None)
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--end-idx', type=int, default=None)
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='vLLM: max sequence length (KV cache). Set to model max (e.g. 32768 for Phi-4-reasoning).',
    )
    parser.add_argument(
        '--gpu-memory-utilization',
        type=float,
        default=0.85,
        help='vLLM GPU memory fraction for KV cache.',
    )
    parser.add_argument('--backend', type=str, default='vllm', choices=['vllm', 'hf'])
    parser.add_argument('--device-map', type=str, default='auto', help='HF: device_map')
    parser.add_argument('--dtype', type=str, default='auto', choices=['auto', 'float16', 'bfloat16'], help='HF: dtype')
    
    args = parser.parse_args()

    if args.backend == "hf":
        batch_inference_hf(
            model_name=args.model,
            input_file=args.input,
            output_dir=args.output,
            n_completions=args.n_completions,
            batch_size=max(1, int(args.batch_size)),
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            system_prompt=args.system_prompt,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            device_map=args.device_map,
            dtype=args.dtype,
        )
    else:
        batch_inference(
            model_name=args.model,
            input_file=args.input,
            output_dir=args.output,
            n_completions=args.n_completions,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            system_prompt=args.system_prompt,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

if __name__ == '__main__':
    main()
