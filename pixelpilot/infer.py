# SPDX-License-Identifier: Apache-2.0
"""Generate PixelPilot responses with a local Qwen2.5-VL SFT/RL checkpoint."""

import argparse
import json
from pathlib import Path

from .io import image_path, read_records, validate_annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local merged Hugging Face checkpoint")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-pixels", type=int, default=720000)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of paper-style sampling")
    parser.add_argument("--reasoning-candidates", help="Condition on teacher reasoning; evaluate only the explicit candidate subset")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if not 0 < args.min_pixels <= args.max_pixels:
        parser.error("Image pixel limits must satisfy 0 < min <= max")
    model_path = Path(args.model)
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError("--model must contain a local merged checkpoint with config.json")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    records = read_records(args.annotations)
    candidates = {}
    if args.reasoning_candidates:
        from .reasoning import read_candidates
        candidates = read_candidates(args.reasoning_candidates)
        if set(candidates) - {row["id"] for row in records}:
            raise ValueError("Reasoning candidate ids absent from annotations")
        records = [row for row in records if row["id"] in candidates]
    if args.limit:
        records = records[:args.limit]
    validate_annotations(records, args.image_root)

    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("7B inference requires a CUDA GPU with the training environment installed")
    set_seed(args.seed)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2", local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True, padding_side="left",
                                             max_pixels=args.max_pixels, min_pixels=args.min_pixels)
    generation = {"max_new_tokens": args.max_new_tokens, "do_sample": not args.greedy}
    if not args.greedy:
        generation.update(temperature=1.0, top_p=0.5, top_k=20)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in records:
            prompt = row["messages"][0]["content"]
            if candidates:
                prompt += "\nAdditional visual reasoning:\n" + candidates[row["id"]]["reasoning"]
            with Image.open(image_path(args.image_root, row["images"][0])) as source:
                image = source.convert("RGB")
            before, after = prompt.split("<image>")
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": before},
                    {"type": "image", "image": image},
                    {"type": "text", "text": after},
                ]},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, **generation)
            response = processor.batch_decode(
                generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )[0]
            result = {"id": row["id"], "predict": response, "label": row["messages"][1]["content"],
                      "generation": {**generation, "seed": args.seed}, "model": model_path.name,
                      "reasoning_conditioned": bool(candidates)}
            result["image_preprocessing"] = {"max_pixels": args.max_pixels, "min_pixels": args.min_pixels}
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
    print(f"Wrote {len(records)} predictions to {output}")


if __name__ == "__main__":
    main()
