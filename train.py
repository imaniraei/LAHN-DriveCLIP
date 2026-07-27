import argparse, json
from contextlib import nullcontext
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from lahn_driveclip.data.dataset import VisionLanguageDataset
from lahn_driveclip.losses.contrastive import symmetric_clip_loss
from lahn_driveclip.models.clip import create_model
from lahn_driveclip.models.lora import lora_state_dict
from lahn_driveclip.utils.config import load_config
from lahn_driveclip.utils.seed import seed_everything

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, preprocess, tokenizer, report = create_model(
        cfg["model"]["name"], cfg["model"]["pretrained"], cfg["lora"]
    )
    model.to(device)
    ratio = 100 * report.trainable_parameters / report.total_parameters
    print(f"LoRA modules: {len(report.replaced_modules)}")
    print(f"Trainable parameters: {report.trainable_parameters:,} ({ratio:.4f}%)")

    dataset = VisionLanguageDataset(
        cfg["data"]["train_manifest"], preprocess, tokenizer,
        cfg["data"].get("image_root")
    )
    loader = DataLoader(
        dataset, batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True, drop_last=True,
        num_workers=int(cfg["training"].get("num_workers", 4)),
        pin_memory=True
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.01))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(int(cfg["training"]["epochs"])):
        model.train()
        running = 0.0
        for step, batch in enumerate(loader, 1):
            images = batch["image"].to(device, non_blocking=True)
            texts = batch["text"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            amp = (torch.autocast("cuda", dtype=torch.float16)
                   if device.type == "cuda" else nullcontext())
            with amp:
                out = model(images, texts)
                loss = symmetric_clip_loss(
                    out["logits_per_image"], out["logits_per_text"]
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, float(cfg["training"].get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
            if step % int(cfg["training"].get("log_every", 20)) == 0:
                print(f"epoch={epoch+1} step={step} loss={running/step:.6f}")

        epoch_loss = running / max(len(loader), 1)
        history.append({"epoch": epoch+1, "train_loss": epoch_loss})
        state = {
            "epoch": epoch, "model": model.state_dict(),
            "lora": lora_state_dict(model),
            "optimizer": optimizer.state_dict(), "config": cfg
        }
        torch.save(state, output_dir / "last.pt")
        torch.save(state["lora"], output_dir / f"lora_epoch_{epoch+1:03d}.pt")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"epoch={epoch+1} train_loss={epoch_loss:.6f}")

if __name__ == "__main__":
    main()
