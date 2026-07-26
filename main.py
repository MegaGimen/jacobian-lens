import os
import torch
import transformers
import jlens
from jlens.examples import load_wikitext_prompts

def top5(logits, tokenizer):
    return [tokenizer.decode([t]) for t in logits.topk(5).indices]

def main():
    jlens.configure_logging()
    MODEL_NAME = "Qwen/Qwen2.5-1.5B"
    LENS_PATH = "qwen2.5_1.5b_jacobian_lens.pt"
    
    # --- 1. 加载模型 ---
    print(f"Loading {MODEL_NAME}...")
    
    # 自动检测并优先使用 CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 指定 torch_dtype="auto" 确保它以半精度（16-bit）加载，极大节省显存
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto"
    ).to(device)
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
    print("Or enter ANY TEXT to change the current prompt.")
    print(f"Available positions: 0 to {len(input_ids)-1} (or negative indexing like -1, -2)")
    print("Type 'q' or 'quit' to exit.")
    print("="*50)

    while True:
        try:
            user_input = input("\nEnter <position> or <new prompt>: ").strip()
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
            
            if not user_input:
                continue
                
            # 判断输入是位置(数字)还是新句子(文本)
            is_number = False
            try:
                pos = int(user_input)
                is_number = True
            except ValueError:
                pass
                
            if not is_number:
                # 认为输入的是新的 prompt
                prompt = user_input
                print(f"\n[Prompt Updated] => '{prompt}'")
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
                print("Tokens and their positions:")
                for i, token_id in enumerate(input_ids):
                    print(f"  [{i}]: {repr(tokenizer.decode([token_id.item()]))}")
                continue
            
            # 以下为输入数字（position）的处理逻辑
            if not (-len(input_ids) <= pos < len(input_ids)):
                print(f"Error: position must be between {-len(input_ids)} and {len(input_ids)-1}")
                continue
                
            # 探测 0 到 n_layers-2 (由于之前报过错，我们确保绝对不触及最后一层)
            all_layers = list(range(model.n_layers - 1))
            
            # 执行透镜应用
            jlens_logits, _, _ = lens.apply(
                model, prompt, layers=all_layers, positions=[pos]
            )
            
            # 解析 token 字符串
            actual_pos = pos if pos >= 0 else len(input_ids) + pos
            token_str = tokenizer.decode([input_ids[actual_pos].item()])
            
            # 遍历每一层输出结果
            print("\n--- 逐层透镜输出 ---")
            for layer in all_layers:
                predictions = top5(jlens_logits[layer][0], tokenizer)
                print(f"position={pos}, layer={layer:>2}, token={repr(token_str)}: {predictions}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            # 彻底暴露所有底层报错细节，方便排查
            import traceback
            traceback.print_exc()
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
