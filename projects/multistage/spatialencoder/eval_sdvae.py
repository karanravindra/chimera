"""Evaluate an off-the-shelf Stable Diffusion VAE on the AFHQ eval set, using the
EXACT same reconstruction-FID protocol as dcae.py / stage1.py so the number is
directly comparable to the multistage leaderboard (v7-lowgan-256 = 5.49 @256px,
v7-champ-gan = 6.54 @128px, 12x-compression trained AEs).

Protocol (matched to dcae.py `evaluate`):
  * AFHQ test split via AFHQDataModule, batch 32, drop_last=True, at --eval-res.
  * Recon = vae.decode(vae.encode(x).latent_dist.mode())  (posterior mean, no noise).
  * rFID = FrechetInceptionDistance(feature=2048, normalize=True), fp32 inception,
    real = imgs.clamp(0,1), fake = recon.clamp(0,1).
  * Also reports val PSNR (MSE-based) and LPIPS-VGG, as dcae.py does.

CAVEAT (printed at the end): the SD VAE is 8x spatial / 4 latent channels =>
~48x compression for RGB, vs the multistage AE's 8x / 16ch = 12x. The SD VAE
compresses 4x harder, so this is the standard strong baseline, NOT a
compression-matched comparison.
"""

import argparse
import math

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from chimera.data import AFHQDataModule
from chimera.utils.seed import seed_everything


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vae", default="stabilityai/sd-vae-ft-mse",
                   help="HF id of the VAE (default: canonical SD1.x VAE, MSE-finetuned)")
    p.add_argument("--eval-res", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lpips-net", default="vgg", choices=["alex", "vgg", "squeeze"])
    args = p.parse_args()

    seed_everything(0, deterministic=True)
    dev = "cuda"

    dm = AFHQDataModule(batch_size=args.batch_size, image_size=args.eval_res, num_workers=7)
    dm.prepare_data()
    dm.setup("test")
    loader = dm.test_dataloader()

    vae = AutoencoderKL.from_pretrained(args.vae).to(dev).eval()
    vae.requires_grad_(False)

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(dev)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type=args.lpips_net, normalize=True)
    lpips.requires_grad_(False)
    lpips = lpips.to(dev).eval()

    loss_sum = lp_sum = cnt = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch[0].to(dev, dtype=torch.float32).clamp(0, 1)  # [0,1]
            x = imgs * 2 - 1                                          # SD VAE wants [-1,1]
            latent = vae.encode(x).latent_dist.mode()                # posterior mean
            rec = vae.decode(latent).sample                          # [-1,1]
            rec = ((rec + 1) / 2).clamp(0, 1)                        # back to [0,1]

            loss_sum += F.mse_loss(rec, imgs).item() * imgs.size(0)
            lp_sum += lpips.net(rec, imgs, normalize=True).mean().item() * imgs.size(0)
            cnt += imgs.size(0)
            fid.update(imgs, real=True)
            fid.update(rec, real=False)

    rfid = fid.compute().item()
    psnr = 10 * math.log10(1 / (loss_sum / cnt))
    lp = lp_sum / cnt

    # latent shape / compression bookkeeping
    lat_c = vae.config.latent_channels
    ds = 2 ** (len(vae.config.block_out_channels) - 1)
    lat_hw = args.eval_res // ds
    comp = (3 * args.eval_res ** 2) / (lat_c * lat_hw ** 2)

    print(f"\n=== SD VAE reconstruction on AFHQ test ({cnt} imgs @ {args.eval_res}px) ===")
    print(f"vae            : {args.vae}")
    print(f"latent         : {lat_c}ch @ {lat_hw}x{lat_hw} (downsample {ds}x)  ->  {comp:.0f}x compression")
    print(f"val_rfid       : {rfid:.4f}")
    print(f"val_psnr       : {psnr:.4f} dB")
    print(f"val_lpips({args.lpips_net}): {lp:.5f}")
    print(f"\nNOTE: SD VAE compresses ~{comp:.0f}x vs the multistage AE's 12x (4x harder), "
          "so this is a strong-baseline reference, not a compression-matched A/B.")


if __name__ == "__main__":
    main()
