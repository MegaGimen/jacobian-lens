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

    # --- 3. 交互式查询透镜 ---
    prompt = "Fact: The currency used in the country shaped like a boot is"
    print(f"\nPrompt: '{prompt}'")
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    print("\nTokens and their positions:")
    for i, token_id in enumerate(input_ids):
        print(f"  [{i}]: {repr(tokenizer.decode([token_id.item()]))}")
        
    print("\n" + "="*50)
    print("Interactive J-Lens Mode Started!")
    print("Format to enter: <position> (e.g., '10', '-2')")
    print(f"Available positions: 0 to {len(input_ids)-1} (or negative indexing like -1, -2)")
    print("Type 'q' or 'quit' to exit.")
    print("="*50)

    while True:
        try:
            user_input = input("\nEnter <position>: ").strip()
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
            
            if not user_input:
                continue
                
            pos = int(user_input)
            
            if not (-len(input_ids) <= pos < len(input_ids)):
                print(f"Error: position must be between {-len(input_ids)} and {len(input_ids)-1}")
                continue
                
            # 一次性提取模型的所有层
            all_layers = list(range(model.n_layers))
            
            # 执行透镜应用，探测这一位置的所有层
            jlens_logits, _, _ = lens.apply(
                model, prompt, layers=all_layers, positions=[pos]
            )
            
            # 解析 token 字符串
            actual_pos = pos if pos >= 0 else len(input_ids) + pos
            token_str = tokenizer.decode([input_ids[actual_pos].item()])
            
            # 遍历每一层输出结果
            for layer in all_layers:
                predictions = top5(jlens_logits[layer][0], tokenizer)
                # 严格按照要求的格式输出
                print(f"position={pos}, layer={layer}, token={repr(token_str)}: {predictions}")
            
        except ValueError:
            print("Error: Invalid input. Please enter a valid integer.")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
