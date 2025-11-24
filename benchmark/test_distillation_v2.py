import os
import json
import argparse
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import sacrebleu


class ParallelDataset(Dataset):
    """
    Simple Japanese–Chinese parallel dataset for evaluation.

    Each item returns:
        - input_ids: tokenized prompt for the source sentence
        - source_text: raw Japanese sentence
        - target_text: raw Chinese reference sentence
    """

    def __init__(self, ja_file: str, zh_file: str, tokenizer, max_length: int = 512) -> None:
        with open(ja_file, "r", encoding="utf-8") as f_ja:
            self.ja_lines = [line.strip() for line in f_ja]

        with open(zh_file, "r", encoding="utf-8") as f_zh:
            self.zh_lines = [line.strip() for line in f_zh]

        assert len(self.ja_lines) == len(self.zh_lines), (
            "Japanese and Chinese files must have the same number of lines, "
            f"got {len(self.ja_lines)} vs {len(self.zh_lines)}"
        )

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.ja_lines)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ja_text = self.ja_lines[idx]
        zh_text = self.zh_lines[idx]

        # Prompt: same style as training / inference scripts
        prompt = (
            "Translate the following segment into Chinese, use a casual tone, "
            "without additional explanation.\n\n"
            f"{ja_text}"
        )
        messages = [{"role": "user", "content": prompt}]

        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        ).squeeze(0)

        # # Truncate if needed
        # if tokenized.size(0) > self.max_length:
        #     tokenized = tokenized[: self.max_length]

        # # Pad to max_length
        # if tokenized.size(0) < self.max_length:
        #     pad_len = self.max_length - tokenized.size(0)
        #     pad_id = self.tokenizer.pad_token_id
        #     if pad_id is None:
        #         # fall back to eos_token_id if pad is not set
        #         pad_id = self.tokenizer.eos_token_id
        #     padding = torch.full((pad_len,), pad_id, dtype=torch.long)
        #     tokenized = torch.cat([tokenized, padding], dim=0)

        # Truncate if needed
        if tokenized.size(0) > self.max_length:
            tokenized = tokenized[-self.max_length:]

        # Pad to max_length (left PAD)
        if tokenized.size(0) < self.max_length:
            pad_len = self.max_length - tokenized.size(0)
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id
            padding = torch.full((pad_len,), pad_id, dtype=torch.long)
            tokenized = torch.cat([padding, tokenized], dim=0)


        return {
            "input_ids": tokenized,
            "source_text": ja_text,
            "target_text": zh_text,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to stack input_ids and keep texts as lists.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    source_text = [item["source_text"] for item in batch]
    target_text = [item["target_text"] for item in batch]
    return {
        "input_ids": input_ids,
        "source_text": source_text,
        "target_text": target_text,
    }


def extract_translation(gen_text: str, source_text: str) -> str:
    """
    Try to extract only the translation part from the raw generated text.

    Heuristics:
      1. If special marker <|extra_0|> exists, take text after it until <|eos|>.
      2. Else if the source_text appears in the output, take text after it.
      3. Else, strip common special tokens and use the remaining text.
    """
    text = gen_text

    # Case 1: assistant marker
    if "<|extra_0|>" in text:
        parts = text.split("<|extra_0|>")
        if len(parts) > 1:
            text = parts[-1]

    # Cut at EOS marker if present
    if "<|eos|>" in text:
        text = text.split("<|eos|>")[0]

    # Case 2: source text in output (e.g., echoing prompt)
    if source_text in text:
        text = text.split(source_text)[-1]

    # Remove some common leftover special tokens
    for token in ["<|startoftext|>", "<|endoftext|>"]:
        text = text.replace(token, "")

    return text.strip()


def compute_bleu(references: List[str], hypotheses: List[str]) -> float:
    """Compute corpus BLEU using sacrebleu."""
    refs = [references]  # list of reference streams
    bleu = sacrebleu.corpus_bleu(hypotheses, refs)
    return float(bleu.score)


def compute_chrf(references: List[str], hypotheses: List[str]) -> float:
    """Compute corpus chrF."""
    refs = [references]
    chrf = sacrebleu.corpus_chrf(hypotheses, refs)
    return float(chrf.score)


def compute_ter(references: List[str], hypotheses: List[str]) -> float:
    """Compute corpus TER."""
    refs = [references]
    ter = sacrebleu.corpus_ter(hypotheses, refs)
    return float(ter.score)


def evaluate_model(
    model,
    tokenizer,
    test_dataset: ParallelDataset,
    device: torch.device,
    batch_size: int = 8,
    max_new_tokens: int = 256,
) -> Dict[str, List[str]]:
    """
    Run deterministic generation on the test set and collect references/hypotheses.
    """
    model.eval()
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    all_references: List[str] = []
    all_hypotheses: List[str] = []

    print("Evaluating model...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Batches"):
            input_ids = batch["input_ids"].to(device)
            references = batch["target_text"]
            sources = batch["source_text"]

            # Deterministic generation (no sampling)
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy decoding
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            generated_texts = tokenizer.batch_decode(
                outputs, skip_special_tokens=False
            )

            for src, ref, gen_text in zip(sources, references, generated_texts):
                hyp = extract_translation(gen_text, src)
                all_references.append(ref)
                all_hypotheses.append(hyp)

    return {"references": all_references, "hypotheses": all_hypotheses}


def save_examples(
    references: List[str],
    hypotheses: List[str],
    output_file: str,
    num_examples: int = 20,
) -> None:
    """Save a subset of examples for manual inspection."""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Examples of translations:\n")
        f.write("=" * 80 + "\n\n")

        for i in range(min(num_examples, len(references))):
            f.write(f"Example {i + 1}:\n")
            f.write(f"Reference: {references[i]}\n")
            f.write(f"Hypothesis: {hypotheses[i]}\n")
            f.write("-" * 80 + "\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic evaluation of a JP->ZH translation model on WCC-JC."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./distilled_model",
        help="Path to the HF model (teacher or student).",
    )
    parser.add_argument(
        "--test_ja_file",
        type=str,
        required=True,
        help="Path to Japanese test file (one sentence per line).",
    )
    parser.add_argument(
        "--test_zh_file",
        type=str,
        required=True,
        help="Path to Chinese reference test file (one sentence per line).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum input sequence length for prompts.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save metrics and outputs (will be created if not exists).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda', 'cpu'). If not set, auto-detect.",
    )

    args = parser.parse_args()

    # Resolve device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Ensure output directory exists (default ./results)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer & model
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # for decoder-only models, set padding side to left
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        # if there's no pad_token, use eos_token as pad_token
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto" if device.type == "cuda" else None,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    if device.type == "cpu":
        model = model.to(device)

    # Load dataset
    print("Loading test dataset...")
    test_dataset = ParallelDataset(
        args.test_ja_file,
        args.test_zh_file,
        tokenizer,
        max_length=args.max_length,
    )
    print(f"Test set size: {len(test_dataset)}")

    # Evaluate
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        device=device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    # Compute metrics
    bleu_score = compute_bleu(results["references"], results["hypotheses"])
    chrf_score = compute_chrf(results["references"], results["hypotheses"])
    ter_score = compute_ter(results["references"], results["hypotheses"])

    print("\n=== Evaluation Results ===")
    print(f"BLEU: {bleu_score:.2f}")
    print(f"chrF: {chrf_score:.2f}")
    print(f"TER:  {ter_score:.2f}")

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    metrics = {
        "BLEU": bleu_score,
        "chrF": chrf_score,
        "TER": ter_score,
        "num_samples": len(results["references"]),
        "model_path": args.model_path,
        "test_ja_file": args.test_ja_file,
        "test_zh_file": args.test_zh_file,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Save detailed results
    detailed_path = os.path.join(args.output_dir, "detailed_results.json")
    detailed = []
    for ref, hyp in zip(results["references"], results["hypotheses"]):
        detailed.append({"reference": ref, "hypothesis": hyp})
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, ensure_ascii=False, indent=2)

    # Save some examples
    examples_path = os.path.join(args.output_dir, "examples.txt")
    save_examples(
        results["references"],
        results["hypotheses"],
        examples_path,
        num_examples=50,
    )

    print(f"\nResults saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
