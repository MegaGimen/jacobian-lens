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

    # --- 3. 使用透镜进行预测 ---
    prompt = "Fact: The currency used in the country shaped like a boot is"
    print(f"\nApplying lens to prompt: '{prompt}'")
    print("Target token position: -2 (at the word 'boot')")
    
    layers = [
        model.n_layers // 4,
        model.n_layers // 2,
        model.n_layers // 4 * 3,
        model.n_layers - 2,
    ]

    # J-lens
    jlens_logits, model_logits, _ = lens.apply(model, prompt, layers=layers, positions=[-2])
    
    # Vanilla logit lens
    logit_lens, _, _ = lens.apply(
        model, prompt, layers=layers, positions=[-2], use_jacobian=False
    )

    print("\n--- 结果对比 ---")
    for layer in layers:
        print(f"L{layer:>3} logit-lens: {top5(logit_lens[layer][0], tokenizer)}")
        print(f"L{layer:>3} J-lens:     {top5(jlens_logits[layer][0], tokenizer)}")
    
    print(f"model final out: {top5(model_logits[0], tokenizer)}")

if __name__ == "__main__":
    main()
