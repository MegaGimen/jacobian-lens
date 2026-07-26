import os
import torch
import transformers
import jlens
from jlens.examples import load_wikitext_prompts

def top5(logits, tokenizer):
    return [tokenizer.decode([t]) for t in logits.topk(5).indices]

def main():
    jlens.configure_logging()
    MODEL_NAME = "gpt2"
    LENS_PATH = "gpt2_jacobian_lens.pt"
    
    # --- 1. 加载模型 ---
    print(f"Loading {MODEL_NAME}...")
    
    # 自动检测并优先使用 CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = jlens.from_hf(hf_model, tokenizer)
    print("Model loaded successfully.")

    # --- 2. 加载透镜 (若无则训练) ---
    if os.path.exists(LENS_PATH):
        print(f"\nFound existing lens at {LENS_PATH}. Loading...")
        lens = jlens.JacobianLens.from_pretrained(LENS_PATH)
    else:
        print(f"\nPre-fitted lens not found at {LENS_PATH}. Training a new lens...")
        try:
            # 官方默认使用100条语料（论文中用到1000条，100条基本可用）
            prompts = load_wikitext_prompts(n_prompts=100)
        except Exception as e:
            print(f"Failed to load wikitext prompts: {e}")
            # fallback dummy prompts
            prompts = ["The quick brown fox jumps over the lazy dog.", 
                       "Artificial intelligence is transforming the world."] * 50

        lens = jlens.fit(
            model, 
            prompts, 
            dim_batch=32, 
            max_seq_len=64, 
            checkpoint_path="gpt2_ckpt.pt"
        )
        lens.save(LENS_PATH)
        print(f"Lens trained and saved to {LENS_PATH}!")

    # --- 3. 使用最优的 J-lens 观测句子中每个位置的内部状态 ---
    # 使用一个简单易懂的 prompt，看它如何一步步推导出 Paris
    prompt = "The capital of France is"
    print(f"\nApplying J-lens to prompt: '{prompt}'")
    
    # 获取所有的 token 以便逐个分析
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    all_positions = list(range(len(input_ids)))
    
    # 选取一个中间偏后的层（对于12层的GPT-2，这里是 L9），往往是 J-lens 破译意图最清晰的一层
    target_layer = model.n_layers // 4 * 3
    print(f"Observing internal state at Layer {target_layer} using J-lens:")

    # 一次性对整句话的所有位置应用雅可比透镜
    jlens_logits, model_logits, _ = lens.apply(
        model, prompt, layers=[target_layer], positions=all_positions
    )

    print("\n--- 逐词透镜输出 ---")
    for i, pos in enumerate(all_positions):
        token_str = tokenizer.decode([input_ids[pos].item()])
        predictions = top5(jlens_logits[target_layer][i], tokenizer)
        # 严格按照要求的格式输出：position x: token=y:[...]
        print(f"position {pos}: token={repr(token_str)}: {predictions}")

if __name__ == "__main__":
    main()
