"""
superresolution.py
Main script for deep learning-based image super-resolution using LAB color space.
"""

# --- Standard Library Imports ---
import warnings
from pathlib import Path
from io import BytesIO

# --- Third-Party Imports ---
import hydra
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from omegaconf import DictConfig
from skimage import color
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from huggingface_hub import login as hf_login
import lpips
from torchinfo import summary


# --- Utility: LAB Color Conversion ---
def to_lab_tensor(img: Image.Image) -> torch.Tensor:
    """
    Convert a PIL RGB image to a normalized LAB torch tensor.
    Args:
        img: PIL Image in RGB mode.
    Returns:
        torch.Tensor: LAB image, shape (3, H, W), normalized.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    lab = color.rgb2lab(arr)
    l_channel = lab[..., 0:1] / 100.0
    a = lab[..., 1:2] / 128.0
    b = lab[..., 2:3] / 128.0
    lab_norm = np.concatenate([l_channel, a, b], axis=-1)
    return torch.from_numpy(lab_norm.transpose(2, 0, 1).copy()).float()


def to_numpy_img(lab_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized LAB torch tensor to a numpy RGB image.
    Args:
        lab_tensor: torch.Tensor, shape (3, H, W), normalized.
    Returns:
        np.ndarray: RGB image, shape (H, W, 3), values in [0, 1].
    """
    arr = lab_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    l_channel = arr[..., 0] * 100.0
    a = arr[..., 1] * 128.0
    b = arr[..., 2] * 128.0
    lab = np.stack([l_channel, a, b], axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rgb = color.lab2rgb(lab)
    return np.clip(rgb, 0, 1)


# --- Dataset Stream ---
def preprocess_stream(dataset, crop_size: int, scale: int):
    """
    Stream and preprocess images from a dataset, yielding (lowres, highres) LAB tensors.
    Args:
        dataset: Streaming dataset iterator.
        crop_size: Size of high-res crop.
        scale: Downscaling factor.
    Yields:
        Tuple[torch.Tensor, torch.Tensor]: (lowres, highres) LAB tensors.
    """
    lowres_size = crop_size // scale
    crop = transforms.RandomCrop(crop_size)
    for example in dataset:
        img_data = example["image"]
        img = (
            img_data
            if isinstance(img_data, Image.Image)
            else Image.open(BytesIO(img_data)).convert("RGB")
        )
        # If image is too small, resize so shortest side >= crop_size
        if min(img.size) < crop_size:
            scale_factor = crop_size / min(img.size)
            new_size = (
                int(round(img.size[0] * scale_factor)),
                int(round(img.size[1] * scale_factor)),
            )
            img = img.resize(new_size, resample=Image.Resampling.BICUBIC)
        # Crop high-res patch from original (or minimally resized) image
        img_patch = crop(img)
        lab_patch = to_lab_tensor(img_patch)
        # Downsample for low-res
        lab_patch_lowres = torch.nn.functional.interpolate(
            lab_patch.unsqueeze(0),
            size=(lowres_size, lowres_size),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
        yield lab_patch_lowres, lab_patch


class SuperResStream(IterableDataset):
    """
    IterableDataset for streaming super-resolution image pairs.
    """

    def __init__(self, dataset, crop_size: int, scale: int):
        self.dataset = dataset
        self.crop_size = crop_size
        self.scale = scale

    def __iter__(self):
        return preprocess_stream(self.dataset, self.crop_size, self.scale)


# --- Model ---
class DeepSuperResNet(nn.Module):
    """
    Deep CNN for super-resolving the L channel in LAB color space.
    Modified: After the head, add a conv to 128 channels. Body uses 128 channels, fewer layers. Upsample and tail adjusted accordingly.
    """

    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # New conv to 128 channels
            nn.ReLU(inplace=True),
        )
        body_layers = []
        for _ in range(8):  # Fewer layers, all 128 channels
            body_layers.append(nn.Conv2d(128, 128, kernel_size=3, padding=1))
            body_layers.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*body_layers)
        self.upsample = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # 128 -> 512, PixelShuffle(2) -> 128
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=3, padding=1),  # 128 -> 512
            nn.PixelShuffle(2),  # 512 -> 128
            nn.ReLU(inplace=True),
        )
        self.tail = nn.Conv2d(
            128,
            1,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = self.body(x)
        x = self.upsample(x)
        x = self.tail(x)
        return x


# --- Visualization ---
def visualize_progress(
    model: nn.Module,
    lowres: torch.Tensor,
    highres: torch.Tensor,
    num_samples: int,
    device: torch.device,
    loss: float | None = None,
    epoch: int | None = None,
    val_lowres: torch.Tensor = None,
    val_highres: torch.Tensor = None,
):
    """
    Visualize and save progress images for training and validation batches.
    """
    model.eval()
    with torch.no_grad():
        # Split AB (L is not used)
        lowres_AB = lowres[:, 1:3]
        highres_L = highres[:, 0:1]
        # Upsample lowres to highres size before passing to model
        upsampled = torch.nn.functional.interpolate(
            lowres,
            size=highres.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        pred_L = model(upsampled.to(device)).cpu()
        # Upscale AB to highres size
        up_AB = torch.nn.functional.interpolate(
            lowres_AB,
            size=highres_L.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        # Ensure pred_L and up_AB have the same spatial size before concatenation
        if pred_L.shape[-2:] != up_AB.shape[-2:]:
            pred_L = torch.nn.functional.interpolate(
                pred_L,
                size=up_AB.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )
        # Combine predicted L with upscaled AB
        outputs = torch.cat([pred_L, up_AB], dim=1)
        # For validation
        val_outputs = None
        if val_lowres is not None and val_highres is not None:
            val_lowres_AB = val_lowres[:, 1:3]
            val_highres_L = val_highres[:, 0:1]
            val_upsampled = torch.nn.functional.interpolate(
                val_lowres,
                size=val_highres.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )
            val_pred_L = model(val_upsampled.to(device)).cpu()
            val_up_AB = torch.nn.functional.interpolate(
                val_lowres_AB,
                size=val_highres_L.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            if val_pred_L.shape[-2:] != val_up_AB.shape[-2:]:
                val_pred_L = torch.nn.functional.interpolate(
                    val_pred_L,
                    size=val_up_AB.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                )
            val_outputs = torch.cat([val_pred_L, val_up_AB], dim=1)
        n = min(6, lowres.size(0))
        n_val = (
            min(6, val_lowres.size(0))
            if (val_lowres is not None and val_highres is not None)
            else 0
        )
        ncols = n + n_val
        fig, axes = plt.subplots(3, ncols, figsize=(2.5 * ncols, 8))
        if ncols == 1:
            axes = axes[:, np.newaxis]
        for i in range(n):
            axes[0, i].imshow(to_numpy_img(lowres[i]))
            axes[0, i].set_title("Train Low-res", fontsize=10)
            axes[1, i].imshow(to_numpy_img(outputs[i]))
            axes[1, i].set_title("Train Output", fontsize=10)
            axes[2, i].imshow(to_numpy_img(highres[i]))
            axes[2, i].set_title("Train High-res", fontsize=10)
            for row in range(3):
                axes[row, i].axis("off")
        if val_outputs is not None:
            for i in range(n_val):
                col = n + i
                if val_lowres is not None:
                    axes[0, col].imshow(to_numpy_img(val_lowres[i]))
                    axes[0, col].set_title("Val Low-res", fontsize=10)
                axes[1, col].imshow(to_numpy_img(val_outputs[i]))
                axes[1, col].set_title("Val Output", fontsize=10)
                if val_highres is not None:
                    axes[2, col].imshow(to_numpy_img(val_highres[i]))
                    axes[2, col].set_title("Val High-res", fontsize=10)
                for row in range(3):
                    axes[row, col].axis("off")
        plt.suptitle(f"Samples seen: {num_samples}")
        plt.tight_layout()
        save_dir = Path("progress_images")
        save_dir.mkdir(exist_ok=True)
        if loss is not None and epoch is not None:
            save_path = (
                save_dir
                / f"superres_samples_epoch_{epoch:03d}_samples_{num_samples:08d}_loss_{loss:.4f}.png"
            )
        elif loss is not None:
            save_path = (
                save_dir / f"superres_samples_{num_samples:08d}_loss_{loss:.4f}.png"
            )
        else:
            save_path = save_dir / f"superres_samples_{num_samples:08d}.png"
        plt.savefig(str(save_path))
        plt.close(fig)


# --- Training Loop ---
def main_train(cfg: DictConfig):
    """
    Main training loop for super-resolution model.
    """
    if cfg.access_token:
        print("Logging in to Hugging Face Hub...")
        hf_login(token=cfg.access_token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crop_size = 128  # 128x128 high-res patch
    scale = 2  # 64x64 low-res patch
    # Dataset
    try:
        train_dataset = load_dataset(
            "imagenet-1k",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        val_dataset = load_dataset(
            "imagenet-1k",
            split="validation",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        return
    train_stream = SuperResStream(train_dataset, crop_size, scale)
    val_stream = SuperResStream(val_dataset, crop_size, scale)
    train_loader = DataLoader(
        train_stream,
        batch_size=cfg.batch_size,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_stream,
        batch_size=cfg.batch_size,
        num_workers=0,
        pin_memory=True,
    )
    train_iterator = iter(train_loader)
    val_iterator = iter(val_loader)
    # Model
    model = DeepSuperResNet().to(device)
    print("Model summary:")
    print(
        summary(
            model,
            input_size=(cfg.batch_size, 3, 64, 64),  # 3 input channels, 64x64 low-res
            col_names=("input_size", "output_size", "num_params"),
            depth=4,
            row_settings=("var_names",),
        ),
    )
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.adam_lr or 1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
    )
    # Loss
    if cfg.loss_type == "lpips":
        lpips_loss_fn = lpips.LPIPS(net="vgg", spatial=False).to(device)

        def criterion(pred, target):
            pred = 2 * pred - 1
            target = 2 * target - 1
            return lpips_loss_fn(pred, target).mean()
    elif cfg.loss_type == "mse":
        mse_loss_fn = nn.MSELoss()

        def criterion(pred, target):
            return mse_loss_fn(pred, target)
    else:
        raise ValueError(f"Unknown loss_type: {cfg.loss_type}")
    # Resume
    start_epoch, num_samples, best_loss = 0, 0, float("inf")
    if cfg.resume_weights_path:
        checkpoint = torch.load(cfg.resume_weights_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            print(
                f"[Warning] Could not load optimizer state: {e}\nContinuing with a fresh optimizer state.",
            )
        if cfg.adam_lr:
            optimizer.param_groups[0]["lr"] = cfg.adam_lr
        start_epoch = checkpoint.get("epoch", 0) + 1
        num_samples = checkpoint.get("samples_seen", 0)
        print(
            f"Resumed from {cfg.resume_weights_path} at epoch {start_epoch}, samples_seen={num_samples}",
        )
    # Overfit batch
    first_train_batch = next(iter(train_loader))
    train_losses, val_losses = [], []
    visualize_every_n_epochs = getattr(cfg, "visualize_every_n_epochs", 1)
    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        running_loss, batch_count = 0.0, 0
        train_iterator = iter(train_loader)
        last_lowres, last_highres = None, None
        with tqdm(
            total=cfg.train_batches_to_load,
            desc=f"Epoch {epoch} [Train]",
            ncols=100,
        ) as pbar:
            for _ in range(cfg.train_batches_to_load):
                if cfg.overfit_one_batch:
                    lowres, highres = first_train_batch
                else:
                    try:
                        lowres, highres = next(train_iterator)
                    except StopIteration:
                        train_iterator = iter(train_loader)
                        lowres, highres = next(train_iterator)
                if cfg.max_samples and num_samples >= cfg.max_samples:
                    print(f"Reached max_samples={cfg.max_samples}. Stopping training.")
                    return
                # --- Only use L channel for SR, AB for upscaling ---
                lowres = lowres.to(device)
                highres = highres.to(device)
                # Upsample lowres to highres size before passing to model
                upsampled = torch.nn.functional.interpolate(
                    lowres,
                    size=highres.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                )
                # Model predicts only L channel
                output_L = model(upsampled)
                highres_L = highres[:, 0:1]
                # Fix: Ensure output_L and highres_L have the same spatial size
                if output_L.shape[-2:] != highres_L.shape[-2:]:
                    output_L = torch.nn.functional.interpolate(
                        output_L,
                        size=highres_L.shape[-2:],
                        mode="bicubic",
                        align_corners=False,
                    )
                loss = criterion(output_L, highres_L)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                running_loss += loss.item()
                num_samples += cfg.batch_size
                batch_count += 1
                last_lowres, last_highres = (
                    lowres.detach().cpu(),
                    highres.detach().cpu(),
                )
                pbar.set_postfix(loss=loss.item(), samples=num_samples)
                pbar.update(1)
        avg_loss = running_loss / batch_count if batch_count > 0 else float("nan")
        train_losses.append(avg_loss)
        # Validation
        if (
            cfg.validation_batches_to_load is not None
            and cfg.validation_batches_to_load > 0
        ):
            model.eval()
            val_running_loss, val_batch_count = 0.0, 0
            val_iterator = iter(val_loader)
            val_last_lowres, val_last_highres = None, None
            with tqdm(
                total=cfg.validation_batches_to_load,
                desc=f"Epoch {epoch} [Val]",
                ncols=100,
            ) as val_pbar:
                with torch.no_grad():
                    for _ in range(cfg.validation_batches_to_load):
                        try:
                            val_lowres, val_highres = next(val_iterator)
                        except StopIteration:
                            val_iterator = iter(val_loader)
                            val_lowres, val_highres = next(val_iterator)
                        val_lowres = val_lowres.to(device)
                        val_highres = val_highres.to(device)
                        val_upsampled = torch.nn.functional.interpolate(
                            val_lowres,
                            size=val_highres.shape[-2:],
                            mode="bicubic",
                            align_corners=False,
                        )
                        val_output_L = model(val_upsampled)
                        val_highres_L = val_highres[:, 0:1]
                        if val_output_L.shape[-2:] != val_highres_L.shape[-2:]:
                            val_output_L = torch.nn.functional.interpolate(
                                val_output_L,
                                size=val_highres_L.shape[-2:],
                                mode="bicubic",
                                align_corners=False,
                            )
                        val_loss = criterion(val_output_L, val_highres_L)
                        val_running_loss += val_loss.item()
                        val_batch_count += 1
                        val_last_lowres, val_last_highres = (
                            val_lowres.detach().cpu(),
                            val_highres.detach().cpu(),
                        )
                        val_pbar.set_postfix(val_loss=val_loss.item())
                        val_pbar.update(1)
            val_avg_loss = (
                val_running_loss / val_batch_count
                if val_batch_count > 0
                else float("nan")
            )
            val_losses.append(val_avg_loss)
            print(
                f"Epoch: {epoch}, Train Loss: {avg_loss:.4f}, Val Loss: {val_avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}\n",
            )
            scheduler.step(val_avg_loss)
            # Visualization logic
            do_visualize = epoch % visualize_every_n_epochs == 0
            if (
                do_visualize
                and last_lowres is not None
                and last_highres is not None
                and val_last_lowres is not None
                and val_last_highres is not None
            ):
                visualize_progress(
                    model,
                    last_lowres,
                    last_highres,
                    num_samples,
                    device,
                    loss=avg_loss,
                    epoch=epoch,
                    val_lowres=val_last_lowres,
                    val_highres=val_last_highres,
                )
        else:
            val_avg_loss = float("nan")
            print(
                f"Epoch: {epoch}, Train Loss: {avg_loss:.4f}, Val Loss: N/A, LR: {optimizer.param_groups[0]['lr']:.2e}",
            )
            # Visualize only training batch if validation is skipped
            if (
                (epoch % visualize_every_n_epochs == 0)
                and last_lowres is not None
                and last_highres is not None
            ):
                visualize_progress(
                    model,
                    last_lowres,
                    last_highres,
                    num_samples,
                    device,
                    loss=avg_loss,
                    epoch=epoch,
                )
        # Save loss curve
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss", marker="o")
        plt.plot(val_losses, label="Validation Loss", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.semilogy()
        plt.legend()
        plt.title("Train/Validation Loss Curve")
        plt.tight_layout()
        plt.savefig("progress_images/loss_curve.png")
        plt.close()
        # Save best weights
        if val_avg_loss < best_loss:
            best_loss = val_avg_loss
            weights_dir = Path("weights")
            weights_dir.mkdir(exist_ok=True)
            best_weights_path = (
                weights_dir
                / f"superres_best_epoch_{epoch:03d}_samples_{num_samples:08d}_loss_{best_loss:.4f}.pth"
            )
            torch.save(
                {
                    "samples_seen": num_samples,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_avg_loss,
                    "epoch": epoch,
                },
                str(best_weights_path),
            )
            print(
                f"[Checkpoint] Saved new best weights at {best_weights_path} (val_loss {best_loss:.4f})",
            )
    print("Training complete.")


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Hydra entry point."""
    main_train(cfg)


if __name__ == "__main__":
    main()
