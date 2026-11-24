#!/usr/bin/env python3
"""
Fine-tune the whole pretrained nvidia/slu_conformer_transformer_large_slurp (encoder + decoder + tokenizer)
on Household-SLU.

Why not NeMo's run_speech_intent_slot_train.py: that recipe builds a new model from the YAML config,
initialises only the encoder from an ASR checkpoint and trains the decoder from scratch. Here the complete
SLURP model is loaded and training continues from it, keeping its SentencePiece tokenizer.

  pip install "nemo_toolkit[asr]==2.7.3"   # NeMo 3.0 dropped the SLU model class
  python scripts/finetune_household.py --epochs 30 --batch_size 32 --lr 1e-4
  # variants
  python scripts/finetune_household.py --freeze_encoder           # train decoder only (fast, less forgetting)
  python scripts/finetune_household.py --replay_manifest /data/slurp/train_slu_subset.json   # keep SLURP

Outputs: nemo_experiments/<name>/... checkpoints and logs, plus <name>.nemo (final model).
"""
import argparse, copy, json
from pathlib import Path

import torch
from omegaconf import open_dict
import nemo.collections.asr as nemo_asr
from nemo.utils.exp_manager import exp_manager
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nemo_compat  # noqa: E402
nemo_compat.apply()  # fixes "'MultiLayerPerceptron' object has no attribute 'mlp'" on NeMo 2.x

try:
    import lightning.pytorch as pl
except ImportError:  # older NeMo
    import pytorch_lightning as pl

ROOT = Path(__file__).resolve().parents[1]


def check_tokenizer(model, manifests):
    """Every target string must be encodable with the SLURP tokenizer (no <unk>, exact round trip)."""
    tok = model.tokenizer
    unk = tok.tokenizer.unk_id() if hasattr(tok, "tokenizer") and hasattr(tok.tokenizer, "unk_id") else None
    bad, lens = [], []
    for m in manifests:
        for line in open(m):
            text = json.loads(line)["text"]
            ids = tok.text_to_ids(text)
            lens.append(len(ids))
            if (unk is not None and unk in ids) or tok.ids_to_text(ids).replace(" ", "") != text.replace(" ", ""):
                bad.append(text)
    print(f"tokenizer check: {len(lens)} targets, mean {sum(lens) / len(lens):.1f} tokens, max {max(lens)}, "
          f"problems: {len(bad)}")
    for b in sorted(set(bad))[:10]:
        print("   cannot encode exactly:", b)
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", default="slu_conformer_transformer_large_slurp",
                    help="NGC/HF model name, or a local .nemo path")
    ap.add_argument("--train", default=str(ROOT / "manifests/train_slu.json"))
    ap.add_argument("--val", default=str(ROOT / "manifests/devel_slu.json"))
    ap.add_argument("--test", default=str(ROOT / "manifests/test_slu.json"))
    ap.add_argument("--replay_manifest", default=None, help="optional SLURP manifest mixed into training")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--accumulate", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--precision", default="bf16-mixed", help="bf16-mixed, 16-mixed or 32")
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--name", default="household_slu_ft")
    ap.add_argument("--exp_dir", default="nemo_experiments")
    ap.add_argument("--monitor", default="val_loss")
    a = ap.parse_args()

    if a.pretrained.endswith(".nemo"):
        model = nemo_asr.models.SLUIntentSlotBPEModel.restore_from(a.pretrained)
    else:
        model = nemo_asr.models.SLUIntentSlotBPEModel.from_pretrained(model_name=a.pretrained)
    if not check_tokenizer(model, [a.train, a.val, a.test]):
        raise SystemExit("Some targets cannot be encoded with the pretrained tokenizer; fix the labels first.")

    trainer = pl.Trainer(devices=a.devices, accelerator="gpu", max_epochs=a.epochs, precision=a.precision,
                         accumulate_grad_batches=a.accumulate, gradient_clip_val=1.0, log_every_n_steps=50,
                         val_check_interval=1.0, logger=False, enable_checkpointing=False,
                         strategy="ddp" if a.devices > 1 else "auto")
    exp_manager(trainer, {"exp_dir": a.exp_dir, "name": a.name, "create_tensorboard_logger": True,
                          "resume_if_exists": True, "resume_ignore_no_checkpoint": True,
                          "checkpoint_callback_params": {"monitor": a.monitor, "mode": "min", "save_top_k": 5,
                                                         "always_save_nemo": True}})
    model.set_trainer(trainer)

    cfg = copy.deepcopy(model.cfg)
    train_manifests = [a.train] + ([a.replay_manifest] if a.replay_manifest else [])
    with open_dict(cfg):
        for key, path, shuffle in (("train_ds", train_manifests, True), ("validation_ds", a.val, False),
                                   ("test_ds", a.test, False)):
            ds = cfg[key]
            ds.manifest_filepath = path
            ds.batch_size = a.batch_size
            ds.shuffle = shuffle
            ds.num_workers = a.num_workers
            ds.use_start_end_token = True  # required for SLU models
            ds.is_tarred = False
            ds.tarred_audio_filepaths = None
        cfg.optim.lr = a.lr
        if "sched" in cfg.optim and cfg.optim.sched is not None:
            cfg.optim.sched.warmup_steps = a.warmup_steps
            cfg.optim.sched.pop("warmup_ratio", None)
            cfg.optim.sched.min_lr = a.lr / 50
    model.cfg = cfg
    model.setup_training_data(cfg.train_ds)
    model.setup_multiple_validation_data(cfg.validation_ds)
    model.setup_multiple_test_data(cfg.test_ds)
    model.setup_optimization(cfg.optim)

    if a.freeze_encoder:
        model.encoder.freeze()
        print("encoder frozen")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_train / 1e6:.1f} M")

    trainer.fit(model)
    trainer.test(model)
    out = Path(a.exp_dir) / f"{a.name}.nemo"
    model.save_to(str(out))
    print("saved", out)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
