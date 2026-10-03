"""
DeepWriting Quickstart Example
Demonstrating plug-and-play Python library usage:
  1. Context manager (automatic model unload)
  2. Explicit model lifecycle (load -> synthesize -> unload)
"""
import os
import deepwriting


def run_context_manager_example():
    print("=== Example 1: Context Manager (Auto-Unload) ===")
    
    # Model is loaded once and automatically freed when exiting the 'with' block
    with deepwriting.load_model() as writer:
        text = "DeepWriting loaded with context manager!"
        print(f"Synthesizing: \"{text}\"")
        
        result = writer.synthesize(text, style=107)
        
        # Save output
        os.makedirs("output", exist_ok=True)
        result.save("output/example_context_manager.png")
        result.save("output/example_context_manager.svg")
        print("Saved to output/example_context_manager.png & .svg")
        
        # Get in-memory PIL Image (no disk I/O required)
        pil_img = result.to_image()
        print(f"In-memory PIL Image size: {pil_img.size}")

    print("Model successfully and automatically unloaded!\n")


def run_explicit_lifecycle_example():
    print("=== Example 2: Explicit Lifecycle (Load -> Synthesize -> Unload) ===")
    
    # 1. Load model into memory
    print("Loading model...")
    writer = deepwriting.load_model()
    
    # 2. Synthesize multiple sentences without reloading the model
    sentences = [
        ("Hello, this is fast generation.", 226),
        ("No model reloading between sentences.", 696)
    ]
    
    for idx, (sentence, style_id) in enumerate(sentences):
        print(f"Synthesizing [{idx+1}/{len(sentences)}]: \"{sentence}\"")
        res = writer.synthesize(sentence, style=style_id)
        res.save(f"output/example_sentence_{idx+1}.png")
        print(f"Generated {len(res)} stroke points -> output/example_sentence_{idx+1}.png")
    
    # 3. Unload model and free RAM
    print("Unloading model...")
    writer.unload()
    print("Model unloaded and memory freed!\n")


if __name__ == "__main__":
    run_context_manager_example()
    run_explicit_lifecycle_example()
