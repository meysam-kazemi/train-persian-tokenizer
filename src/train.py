from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import yaml
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, normalizers
from tokenizers import pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm


ARABIC_PERSIAN_TRANSLATION = str.maketrans({
    "ي": "ی",
    "ى": "ی",
    "ك": "ک",
})


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "training.log", encoding="utf-8"),
        ],
        force=True,
    )


def normalize_persian(text: str) -> str:
    """Conservative Persian normalization; preserves ZWNJ and punctuation."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(ARABIC_PERSIAN_TRANSLATION)
    text = text.replace("ـ", "")  # Arabic tatweel
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(example: dict[str, Any], fields: list[str]) -> str:
    for field in fields:
        value = example.get(field)
        if isinstance(value, str) and value.strip():
            return value
    # Fallback for datasets whose schema differs from the expected one.
    for value in example.values():
        if isinstance(value, str) and value.strip():
            return value
    return ""


def valid_text(text: str, cfg: dict[str, Any]) -> bool:
    if len(text) < cfg["min_characters"]:
        return False
    if len(text) > cfg["max_characters"]:
        text = text[:cfg["max_characters"]]
    persian_count = len(re.findall(r"[\u0600-\u06FF]", text))
    if persian_count < cfg["min_persian_characters"]:
        return False
    ratio = persian_count / max(len(text), 1)
    if ratio < cfg["min_persian_ratio"]:
        return False
    words = text.split()
    if len(words) >= 30 and len(set(words)) / len(words) < 0.12:
        return False
    return True


def load_one_source(source: dict[str, Any], shuffle_buffer: int, seed: int):
    """Try configured dataset configs in order; report failures and continue."""
    errors = []
    configs = source.get("configs", [None])
    if not configs:
        configs = [None]

    for config_name in configs:
        try:
            kwargs = {
                "path": source["name"],
                "split": source.get("split", "train"),
                "streaming": True,
            }
            if config_name is not None:
                kwargs["name"] = config_name

            logging.info(
                "Loading %s (config=%s, split=%s)",
                source["name"], config_name, source.get("split", "train")
            )
            ds = load_dataset(**kwargs)
            if shuffle_buffer and hasattr(ds, "shuffle"):
                ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
            return ds, config_name
        except Exception as exc:
            errors.append(f"config={config_name}: {type(exc).__name__}: {exc}")
            logging.warning("Could not load %s: %s", source["name"], exc)

    logging.error("Skipping %s. Attempts: %s", source["name"], " || ".join(errors))
    return None, None


def build_corpus(config: dict[str, Any], corpus_path: Path) -> dict[str, Any]:
    data_cfg = config["data"]
    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    stats: list[dict[str, Any]] = []
    total = 0
    total_chars = 0

    with corpus_path.open("w", encoding="utf-8") as output:
        for source in data_cfg["datasets"]:
            if not source.get("enabled", True):
                continue
            if total >= data_cfg["max_total_documents"]:
                break

            ds, selected_config = load_one_source(
                source,
                data_cfg.get("shuffle_buffer", 0),
                config["project"]["seed"],
            )
            row = {
                "name": source["name"],
                "selected_config": selected_config,
                "read": 0,
                "accepted": 0,
                "duplicates": 0,
                "rejected": 0,
                "error": None,
            }
            if ds is None:
                row["error"] = "Could not load any configured dataset config"
                stats.append(row)
                continue

            source_limit = int(source.get("max_documents", data_cfg["max_total_documents"]))
            fields = source.get("text_fields", ["text"])

            try:
                for example in tqdm(ds, desc=source["name"], unit="docs"):
                    if total >= data_cfg["max_total_documents"]:
                        break
                    if row["read"] >= source_limit:
                        break

                    row["read"] += 1
                    raw = extract_text(example, fields)
                    text = normalize_persian(raw)

                    if len(text) > data_cfg["max_characters"]:
                        text = text[:data_cfg["max_characters"]].rstrip()

                    if not valid_text(text, data_cfg):
                        row["rejected"] += 1
                        continue

                    digest = hashlib.blake2b(
                        text.encode("utf-8"), digest_size=16
                    ).hexdigest()

                    if data_cfg.get("deduplicate", True) and digest in seen:
                        row["duplicates"] += 1
                        continue

                    seen.add(digest)
                    # One document per line; remove internal line breaks so the
                    # tokenizer iterator receives complete document strings.
                    output.write(re.sub(r"\s+", " ", text) + "\n")
                    row["accepted"] += 1
                    total += 1
                    total_chars += len(text)

            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                logging.exception("Error while iterating %s", source["name"])

            logging.info(
                "%s: read=%d accepted=%d duplicate=%d rejected=%d",
                source["name"], row["read"], row["accepted"],
                row["duplicates"], row["rejected"]
            )
            stats.append(row)

    summary = {
        "total_documents": total,
        "total_characters": total_chars,
        "average_characters_per_document": total_chars / max(total, 1),
        "sources": stats,
    }
    return summary


def corpus_iterator(corpus_path: Path) -> Iterator[str]:
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def train_tokenizer(config: dict[str, Any], corpus_path: Path, output_dir: Path):
    tok_cfg = config["tokenizer"]
    special_tokens = tok_cfg["special_tokens"]

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # NFC is safe for canonical Unicode equivalence. Arabic Yeh/Kaf are
    # normalized consistently with the corpus preprocessing.
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC(),
        normalizers.Replace("ي", "ی"),
        normalizers.Replace("ى", "ی"),
        normalizers.Replace("ك", "ک"),
    ])

    if tok_cfg.get("byte_level", True):
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=tok_cfg.get("add_prefix_space", False)
        )
        tokenizer.decoder = decoders.ByteLevel()
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
    else:
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        initial_alphabet = []

    trainer = trainers.BpeTrainer(
        vocab_size=int(tok_cfg["vocab_size"]),
        min_frequency=int(tok_cfg["min_frequency"]),
        special_tokens=special_tokens,
        initial_alphabet=initial_alphabet,
        show_progress=True,
    )

    logging.info("Starting BPE training. Target vocabulary=%s", tok_cfg["vocab_size"])
    tokenizer.train_from_iterator(
        corpus_iterator(corpus_path),
        trainer=trainer,
        length=None,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    backend_path = output_dir / "tokenizer.json"
    tokenizer.save(str(backend_path))

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        clean_up_tokenization_spaces=False,
        model_max_length=2048,
    )
    hf_tokenizer.save_pretrained(output_dir)

    logging.info("Saved tokenizer to %s", output_dir)
    return hf_tokenizer


def evaluate_tokenizer(config: dict[str, Any], tokenizer, output_dir: Path):
    texts = config["evaluation"]["test_texts"]
    rows = []

    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        rows.append({
            "text": text,
            "token_count": len(ids),
            "whitespace_word_count": len(text.split()),
            "tokens": tokens,
            "round_trip_exact": decoded == text,
            "decoded": decoded,
        })

    report = {
        "vocab_size": len(tokenizer),
        "special_tokens": tokenizer.special_tokens_map,
        "examples": rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "evaluation.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("TOKENIZER EVALUATION")
    print("=" * 72)
    print("Vocabulary size:", len(tokenizer))
    for row in rows:
        print("\nText:", row["text"])
        print("Token count:", row["token_count"])
        print("Tokens:", row["tokens"])
        print("Round-trip exact:", row["round_trip_exact"])

    logging.info("Evaluation saved to %s", report_path)


def compare_with_qwen(config: dict[str, Any], new_tokenizer, output_dir: Path):
    eval_cfg = config["evaluation"]
    if not eval_cfg.get("compare_qwen", False):
        return

    from transformers import AutoTokenizer

    model_name = eval_cfg.get("qwen_model", "Qwen/Qwen2.5-7B-Instruct")
    logging.info("Loading comparison tokenizer: %s", model_name)
    qwen = AutoTokenizer.from_pretrained(model_name)

    rows = []
    for text in eval_cfg["test_texts"]:
        qwen_ids = qwen.encode(text, add_special_tokens=False)
        new_ids = new_tokenizer.encode(text, add_special_tokens=False)
        rows.append({
            "text": text,
            "qwen_tokens": len(qwen_ids),
            "new_tokens": len(new_ids),
            "reduction_percent": (
                100.0 * (len(qwen_ids) - len(new_ids)) / max(len(qwen_ids), 1)
            ),
        })

    path = output_dir / "qwen_comparison.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nQWEN COMPARISON")
    for row in rows:
        print(
            f'{row["text"]}\n'
            f'  Qwen={row["qwen_tokens"]} | '
            f'New={row["new_tokens"]} | '
            f'Reduction={row["reduction_percent"]:.1f}%'
        )


def main():
    parser = argparse.ArgumentParser(description="Train a Persian BPE tokenizer.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--rebuild-corpus", action="store_true",
                        help="Download/process datasets again, even if corpus exists.")
    parser.add_argument("--compare-qwen", action="store_true",
                        help="Compare token counts with the configured Qwen tokenizer.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    random.seed(config["project"]["seed"])

    output_dir = Path(config["project"]["output_dir"])
    data_dir = Path(config["project"]["data_dir"])
    log_dir = Path(config["project"]["log_dir"])
    setup_logging(log_dir)

    corpus_path = data_dir / "persian_corpus.txt"
    stats_path = data_dir / "dataset_statistics.json"

    if args.rebuild_corpus or not corpus_path.exists():
        logging.info("Building corpus from Hugging Face datasets...")
        summary = build_corpus(config, corpus_path)
        data_dir.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logging.info("Corpus summary: %s", summary)
    else:
        logging.info("Using existing corpus: %s", corpus_path)

    if not corpus_path.exists() or corpus_path.stat().st_size == 0:
        raise RuntimeError(
            "No corpus was produced. Check dataset IDs/configs, network access, "
            "and logs/training.log."
        )

    tokenizer = train_tokenizer(config, corpus_path, output_dir)
    evaluate_tokenizer(config, tokenizer, output_dir)

    if args.compare_qwen:
        config["evaluation"]["compare_qwen"] = True
    compare_with_qwen(config, tokenizer, output_dir)

    logging.info("Project completed successfully.")


if __name__ == "__main__":
    main()
