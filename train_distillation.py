import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup
)
from transformers import AutoConfig
import json
from tqdm import tqdm
import argparse
from dataclasses import dataclass
from typing import Optional
import bisect
import numpy as np
from itertools import islice


class ParallelDataset(Dataset):
    """Dataset for Japanese-Chinese parallel texts"""
    def __init__(self, ja_file, zh_file, tokenizer, max_length=512):
        with open(ja_file, 'r', encoding='utf-8') as f:
            self.ja_lines = [line.strip() for line in f.readlines()]
        with open(zh_file, 'r', encoding='utf-8') as f:
            self.zh_lines = [line.strip() for line in f.readlines()]
        
        assert len(self.ja_lines) == len(self.zh_lines), \
            "Japanese and Chinese files must have the same number of lines"
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.ja_lines)
    
    def __getitem__(self, idx):
        ja_text = self.ja_lines[idx]
        zh_text = self.zh_lines[idx]
        
        # Format as translation task using chat template
        # For Japanese to Chinese: using the prompt format from the model
        messages = [
            {"role": "user", "content": f"Translate the following segment into Chinese, use a casual tone, without additional explanation.\n\n{ja_text}"},
            {"role": "assistant", "content": zh_text}
        ]
        
        # Apply chat template and tokenize
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        ).squeeze(0)
        
        # For causal LM, labels are the same as input_ids (with -100 for tokens to ignore)
        # We want to predict all tokens, so labels = input_ids.clone()
        labels = tokenized.clone()
        
        # Find where the assistant response starts (after <|extra_0|>)
        # Mask the prompt part (set to -100 so it's ignored in loss)
        # This is a simplified approach - in practice, you might want more precise masking
        input_len = len(tokenized)
        if input_len > self.max_length:
            tokenized = tokenized[:self.max_length]
            labels = labels[:self.max_length]
        
        # Pad to max_length
        if len(tokenized) < self.max_length:
            padding_length = self.max_length - len(tokenized)
            tokenized = torch.cat([tokenized, torch.full((padding_length,), self.tokenizer.pad_token_id)])
            labels = torch.cat([labels, torch.full((padding_length,), -100)])
        
        return {
            'input_ids': tokenized,
            'labels': labels,
            'source_text': ja_text,
            'target_text': zh_text
        }


class TeacherLogitsDataset(Dataset):
    """Dataset backed by precomputed teacher logits shards"""

    def __init__(self, logits_dir, split="train"):
        metadata_path = os.path.join(logits_dir, f"{split}_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file {metadata_path} not found. "
                "Ensure you have generated teacher logits for this split."
            )

        with open(metadata_path, "r", encoding="utf-8") as meta_file:
            metadata = json.load(meta_file)

        self.logits_dir = logits_dir
        self.files = metadata.get("files", [])
        if not self.files:
            raise ValueError(f"No shard files listed in metadata {metadata_path}")

        self._cumulative_sizes = []
        running_total = 0
        for file_info in self.files:
            num_samples = int(file_info.get("num_samples", 0))
            if num_samples <= 0:
                raise ValueError(f"Shard {file_info} has non-positive num_samples")
            running_total += num_samples
            self._cumulative_sizes.append(running_total)

        self._total_samples = running_total
        self._cache = {}
        self._last_loaded = None

    def __len__(self):
        return self._total_samples

    def _load_shard(self, shard_idx):
        if shard_idx == self._last_loaded and self._last_loaded in self._cache:
            return self._cache[self._last_loaded]

        shard_info = self.files[shard_idx]
        shard_path = os.path.join(self.logits_dir, shard_info["filename"])
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Shard file {shard_path} not found")

        shard_data = torch.load(shard_path, map_location="cpu")
        expected_keys = {"input_ids", "labels", "teacher_logits"}
        if not expected_keys.issubset(shard_data):
            missing = expected_keys - set(shard_data.keys())
            raise KeyError(
                f"Shard {shard_path} missing expected keys: {missing}"
            )

        self._cache = {shard_idx: shard_data}
        self._last_loaded = shard_idx
        return shard_data

    def __getitem__(self, idx):
        if idx < 0 or idx >= self._total_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self._total_samples}")

        shard_idx = bisect.bisect_right(self._cumulative_sizes, idx)
        start_idx = 0 if shard_idx == 0 else self._cumulative_sizes[shard_idx - 1]
        local_idx = idx - start_idx

        shard_data = self._load_shard(shard_idx)

        return {
            "input_ids": shard_data["input_ids"][local_idx],
            "labels": shard_data["labels"][local_idx],
            "teacher_logits": shard_data["teacher_logits"][local_idx],
        }


@dataclass
class DistillationArguments:
    """Arguments for knowledge distillation training"""
    teacher_model_path: str = "Hunyuan-MT-7B"
    student_model_path: Optional[str] = None
    output_dir: str = "./distilled_model"
    teacher_logits_dir: Optional[str] = None
    teacher_logits_split: str = "train"
    
    # Dataset paths
    train_ja_file: str = "Web-Crawled-Corpus-for-Japanese-Chinese-NMT/train data/train-cj-demo-100000.ja.txt"
    train_zh_file: str = "Web-Crawled-Corpus-for-Japanese-Chinese-NMT/train data/train-cj-demo-100000.ch.txt"
    val_ja_file: str = "Web-Crawled-Corpus-for-Japanese-Chinese-NMT/dev data/WCC-JC/valid-ja.txt"
    val_zh_file: str = "Web-Crawled-Corpus-for-Japanese-Chinese-NMT/dev data/WCC-JC/valid-zh.txt"
    
    # Training hyperparameters
    batch_size: int = 4
    learning_rate: float = 5e-5
    num_epochs: int = 3
    max_length: int = 512
    warmup_steps: int = 500
    max_iterations_per_epoch: Optional[int] = None  # Limit iterations per epoch (None = use all data)
    
    # Distillation hyperparameters
    temperature: float = 4.0
    alpha: float = 0.7  # Weight for distillation loss vs hard label loss
    use_soft_labels: bool = True
    
    # Student model config (smaller than teacher)
    student_hidden_size: int = 2048
    student_num_layers: int = 12
    student_num_heads: int = 16
    student_intermediate_size: int = 8192
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    gradient_accumulation_steps: int = 1
    use_amp: bool = True  # Automatic Mixed Precision


class DistillationTrainer:
    """Trainer for knowledge distillation"""
    
    def __init__(self, args: DistillationArguments):
        self.args = args
        self.device = torch.device(args.device)
        
        print("Loading teacher tokenizer...")
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path)
        self.teacher_model = None

        if args.teacher_logits_dir:
            print(f"Using precomputed teacher logits from {args.teacher_logits_dir}")
        else:
            print("Loading teacher model...")
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                args.teacher_model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            self.teacher_model.eval()
        
        print("Creating student model...")
        self.student_tokenizer = self.teacher_tokenizer  # Share tokenizer
        
        # Load teacher config and modify for student
        teacher_config = AutoConfig.from_pretrained(args.teacher_model_path)
        
        if args.student_model_path and os.path.exists(args.student_model_path):
            print(f"Loading student model from {args.student_model_path}")
            self.student_model = AutoModelForCausalLM.from_pretrained(
                args.student_model_path
            )
        else:
            print("Initializing new student model...")
            # Create smaller student model config
            student_config = AutoConfig.from_pretrained(args.teacher_model_path)
            student_config.hidden_size = args.student_hidden_size
            student_config.num_hidden_layers = args.student_num_layers
            student_config.num_attention_heads = args.student_num_heads
            student_config.num_key_value_heads = args.student_num_heads // 4
            student_config.intermediate_size = args.student_intermediate_size
            student_config.attention_head_dim = args.student_hidden_size // args.student_num_heads
            
            self.student_model = AutoModelForCausalLM.from_config(student_config)
            # Initialize student embeddings from teacher (if compatible)
            self._init_student_from_teacher()
        
        # Move student to device
        self.student_model = self.student_model.to(self.device)
        student_config = self.student_model.config
        # Note: AMP scaler works better with FP16 than BFloat16
        # We'll convert to FP16 after initializing AMP scaler if needed

        # Teacher config
        print(f"Teacher config: {teacher_config}")
        print(f"Student config: {student_config}")
        
        print("Loading datasets...")
        if args.teacher_logits_dir:
            self.train_dataset = TeacherLogitsDataset(
                args.teacher_logits_dir,
                split=args.teacher_logits_split
            )
        else:
            self.train_dataset = ParallelDataset(
                args.train_ja_file,
                args.train_zh_file,
                self.student_tokenizer,
                max_length=args.max_length
            )
        
        self.val_dataset = ParallelDataset(
            args.val_ja_file,
            args.val_zh_file,
            self.student_tokenizer,
            max_length=args.max_length
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory if args.device == "cuda" else False,
            prefetch_factor=2 if args.num_workers > 0 else None
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory if args.device == "cuda" else False,
            prefetch_factor=2 if args.num_workers > 0 else None
        )
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=args.learning_rate
        )
        
        total_steps = (len(self.train_loader) // args.gradient_accumulation_steps) * args.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Mixed precision scaler
        # Note: AMP scaler works best with FP16, not BFloat16
        self.use_amp = args.use_amp and args.device == "cuda"
        if self.use_amp:
            # Use FP16 scaler (not BFloat16)
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            # Ensure student model is in FP16 for AMP compatibility (BPFloat16 causes issues)
            self.student_model = self.student_model.half()
        else:
            self.scaler = None
        
    def _init_student_from_teacher(self):
        """Initialize student model from teacher (for compatible layers)"""
        # This is a simplified initialization
        # In practice, you might want more sophisticated initialization
        pass
    
    def get_teacher_logits(self, input_ids):
        """Get teacher model logits for distillation"""
        if self.teacher_model is None:
            raise RuntimeError(
                "Teacher model is not initialized. Provide precomputed teacher logits "
                "or instantiate with a teacher model."
            )
        with torch.no_grad():
            # Move input to teacher device efficiently
            teacher_input = input_ids.to(self.teacher_model.device, non_blocking=True)
            
            # Teacher is already in bfloat16, no need for autocast
            outputs = self.teacher_model(
                input_ids=teacher_input,
                labels=None,
                output_hidden_states=False
            )
            
            # Extract logits (full sequence)
            logits = outputs.logits
            
            return logits
    
    def _prepare_teacher_logits(self, batch, input_ids, target_dtype):
        """Retrieve teacher logits from cache or by querying the teacher model"""
        if not self.args.use_soft_labels:
            return None

        if "teacher_logits" in batch:
            logits = batch["teacher_logits"]
            if not isinstance(logits, torch.Tensor):
                raise TypeError("teacher_logits in batch must be a torch.Tensor")
            logits = logits.to(self.device, non_blocking=True)
        else:
            logits = self.get_teacher_logits(input_ids)
            logits = logits.to(self.device, non_blocking=True)

        if target_dtype is not None and logits.dtype != target_dtype:
            logits = logits.to(target_dtype)

        return logits
    
    def compute_distillation_loss(self, student_logits, teacher_logits, labels, temperature):
        """Compute knowledge distillation loss"""
        # Shift logits for next token prediction
        # Student logits: shape [batch, seq_len, vocab]
        # Teacher logits: shape [batch, seq_len, vocab]
        # Labels: shape [batch, seq_len]
        
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Create mask for valid tokens (ignore padding and -100)
        mask = (shift_labels != -100) & (shift_labels != self.student_tokenizer.pad_token_id)
        mask = mask.float().unsqueeze(-1)  # [batch, seq_len-1, 1]
        
        # Compute KL divergence loss (soft labels)
        student_log_probs = F.log_softmax(shift_student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(shift_teacher_logits / temperature, dim=-1)
        
        # Compute KL divergence
        kl_div = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='none',
            log_target=False
        )  # [batch, seq_len-1, vocab]
        
        kl_div = kl_div.sum(dim=-1)  # [batch, seq_len-1]
        kl_div = kl_div * mask.squeeze(-1)  # Apply mask
        
        # Average over valid tokens
        num_valid_tokens = mask.sum().clamp(min=1.0)
        kl_loss = (temperature ** 2) * kl_div.sum() / num_valid_tokens
        
        # Compute cross-entropy loss (hard labels)
        ce_loss = F.cross_entropy(
            shift_student_logits.view(-1, shift_student_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        # Combine losses
        total_loss = self.args.alpha * kl_loss + (1 - self.args.alpha) * ce_loss
        
        return total_loss, kl_loss, ce_loss
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.student_model.train()
        total_loss = 0
        total_kl_loss = 0
        total_ce_loss = 0
        
        # Determine max iterations
        max_iters = self.args.max_iterations_per_epoch
        if max_iters is not None:
            # Use itertools.islice to limit the iterator
            loader_iter = islice(self.train_loader, max_iters)
            num_batches = min(max_iters, len(self.train_loader))
        else:
            loader_iter = self.train_loader
            num_batches = len(self.train_loader)
        
        progress_bar = tqdm(loader_iter, desc=f"Epoch {epoch+1}", total=num_batches)
        self.optimizer.zero_grad()  # Zero gradients at the start
        
        num_batches_processed = 0
        for batch_idx, batch in enumerate(progress_bar):
            num_batches_processed += 1
            # Move to device with non_blocking for faster transfer
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # Use mixed precision for student forward pass
            if self.use_amp:
                # Use FP16 autocast (BPFloat16 not fully supported by AMP scaler)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    # Forward pass through student
                    student_outputs = self.student_model(
                        input_ids=input_ids,
                        labels=None
                    )
                    student_logits = student_outputs.logits
                    
                    # Get teacher logits and compute loss
                    if self.args.use_soft_labels:
                        teacher_logits = self._prepare_teacher_logits(
                            batch,
                            input_ids,
                            student_logits.dtype
                        )
                        loss, kl_loss, ce_loss = self.compute_distillation_loss(
                            student_logits,
                            teacher_logits,
                            labels,
                            self.args.temperature
                        )
                        total_kl_loss += kl_loss.item()
                        total_ce_loss += ce_loss.item()
                    else:
                        loss = F.cross_entropy(
                            student_logits.view(-1, student_logits.size(-1)),
                            labels.view(-1),
                            ignore_index=-100
                        )
                
                # Scale loss and backward pass
                loss = loss / self.args.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                # Forward pass through student
                student_outputs = self.student_model(
                    input_ids=input_ids,
                    labels=None
                )
                student_logits = student_outputs.logits
                
                # Get teacher logits and compute loss
                if self.args.use_soft_labels:
                    teacher_logits = self._prepare_teacher_logits(
                        batch,
                        input_ids,
                        student_logits.dtype
                    )
                    loss, kl_loss, ce_loss = self.compute_distillation_loss(
                        student_logits,
                        teacher_logits,
                        labels,
                        self.args.temperature
                    )
                    total_kl_loss += kl_loss.item()
                    total_ce_loss += ce_loss.item()
                else:
                    loss = F.cross_entropy(
                        student_logits.view(-1, student_logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100
                    )
                
                loss = loss / self.args.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.args.gradient_accumulation_steps
            
            # Update weights only after accumulating gradients
            if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                avg_loss_so_far = total_loss / (batch_idx + 1)
                postfix = {
                    'loss': loss.item() * self.args.gradient_accumulation_steps,
                    'avg_loss': avg_loss_so_far
                }
                if self.args.use_soft_labels:
                    postfix['kl'] = kl_loss.item()
                    postfix['ce'] = ce_loss.item()
                progress_bar.set_postfix(postfix)
        
        # Handle any remaining accumulated gradients if we break early
        if num_batches_processed > 0 and num_batches_processed % self.args.gradient_accumulation_steps != 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / num_batches_processed if num_batches_processed > 0 else 0.0
        avg_kl_loss = total_kl_loss / num_batches_processed if (self.args.use_soft_labels and num_batches_processed > 0) else 0
        avg_ce_loss = total_ce_loss / num_batches_processed if (self.args.use_soft_labels and num_batches_processed > 0) else 0
        
        return avg_loss, avg_kl_loss, avg_ce_loss
    
    def validate(self):
        """Validate the model"""
        self.student_model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Compute loss manually to ensure consistency
                outputs = self.student_model(
                    input_ids=input_ids,
                    labels=None
                )
                student_logits = outputs.logits
                
                # Shift logits and labels
                shift_logits = student_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.args.num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.args.num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss, kl_loss, ce_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}")
            if self.args.use_soft_labels:
                print(f"  KL Loss: {kl_loss:.4f}")
                print(f"  CE Loss: {ce_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            print(f"Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch-{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            self.student_model.save_pretrained(checkpoint_dir)
            self.student_tokenizer.save_pretrained(checkpoint_dir)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"New best model! Saving to {self.args.output_dir}")
                self.student_model.save_pretrained(self.args.output_dir)
                self.student_tokenizer.save_pretrained(self.args.output_dir)
                
                # Save training config
                config_dict = {
                    'teacher_model': self.args.teacher_model_path,
                    'student_config': {
                        'hidden_size': self.args.student_hidden_size,
                        'num_layers': self.args.student_num_layers,
                        'num_heads': self.args.student_num_heads,
                        'intermediate_size': self.args.student_intermediate_size
                    },
                    'training_args': {
                        'temperature': self.args.temperature,
                        'alpha': self.args.alpha,
                        'learning_rate': self.args.learning_rate,
                        'batch_size': self.args.batch_size,
                        'num_epochs': self.args.num_epochs
                    }
                }
                
                with open(os.path.join(self.args.output_dir, "distillation_config.json"), 'w') as f:
                    json.dump(config_dict, f, indent=2)
        
        print("\nTraining completed!")


def main():
    parser = argparse.ArgumentParser(description="Train distilled model")
    parser.add_argument("--teacher_model_path", type=str, default="Hunyuan-MT-7B",
                        help="Path to teacher model")
    parser.add_argument("--student_model_path", type=str, default=None,
                        help="Path to student model (if continuing training)")
    parser.add_argument("--output_dir", type=str, default="./distilled_model",
                        help="Output directory for distilled model")
    parser.add_argument("--teacher_logits_dir", type=str, default=None,
                        help="Directory containing precomputed teacher logits")
    parser.add_argument("--teacher_logits_split", type=str, default="train",
                        help="Split name to load from teacher logits directory")
    parser.add_argument("--train_ja_file", type=str,
                        default="Web-Crawled-Corpus-for-Japanese-Chinese-NMT/WCC-JC 2.0/train-ja-demo-200k.txt")
    parser.add_argument("--train_zh_file", type=str,
                        default="Web-Crawled-Corpus-for-Japanese-Chinese-NMT/WCC-JC 2.0/train-ch-demo-200k.txt")
    parser.add_argument("--val_ja_file", type=str,
                        default="Web-Crawled-Corpus-for-Japanese-Chinese-NMT/dev data/WCC-JC/valid-ja.txt")
    parser.add_argument("--val_zh_file", type=str,
                        default="Web-Crawled-Corpus-for-Japanese-Chinese-NMT/dev data/WCC-JC/valid-zh.txt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=2)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--student_hidden_size", type=int, default=1024)
    parser.add_argument("--student_num_layers", type=int, default=8)
    parser.add_argument("--student_num_heads", type=int, default=8)
    parser.add_argument("--student_intermediate_size", type=int, default=8192)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Use automatic mixed precision training")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false",
                        help="Disable automatic mixed precision training")
    parser.add_argument("--max_iterations_per_epoch", type=int, default=2500,
                        help="Maximum number of iterations per epoch (None = use all data)")
    
    args = parser.parse_args()
    
    dist_args = DistillationArguments(
        teacher_model_path=args.teacher_model_path,
        student_model_path=args.student_model_path,
        output_dir=args.output_dir,
        teacher_logits_dir=args.teacher_logits_dir,
        teacher_logits_split=args.teacher_logits_split,
        train_ja_file=args.train_ja_file,
        train_zh_file=args.train_zh_file,
        val_ja_file=args.val_ja_file,
        val_zh_file=args.val_zh_file,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        temperature=args.temperature,
        alpha=args.alpha,
        student_hidden_size=args.student_hidden_size,
        student_num_layers=args.student_num_layers,
        student_num_heads=args.student_num_heads,
        student_intermediate_size=args.student_intermediate_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=args.use_amp,
        max_iterations_per_epoch=args.max_iterations_per_epoch
    )
    
    trainer = DistillationTrainer(dist_args)
    trainer.train()


if __name__ == "__main__":
    main()
