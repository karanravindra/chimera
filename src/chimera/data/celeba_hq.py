"""CelebA-HQ DataModule (StarGAN v2 download + materialized uint8 + device-side cast).

High-quality aligned face crops (female/male). Fetched via StarGAN v2's
``download.sh`` Dropbox zip; see :class:`chimera.data.base.StarGANImageDataModule`
for the materialization rationale and :class:`chimera.data.base.ImageDataModule`
for the throughput one.
"""

from chimera.data.base import StarGANImageDataModule


class CelebAHQDataModule(StarGANImageDataModule):
    url = "https://www.dropbox.com/scl/fi/s3tr5yv8a930gfqfgdtn9/celeba_hq.zip?rlkey=xlv3qjl8zskg3usfruj00dm6j"
    archive_name = "celeba_hq.zip"
    dirname = "celeba_hq"


if __name__ == "__main__":
    import time

    dm = CelebAHQDataModule(batch_size=256)
    dm.prepare_data()
    dm.setup("fit")
    train_loader = dm.train_dataloader()

    n_batches = 100
    start_time = time.time()
    for i, batch in enumerate(train_loader):
        if i + 1 >= n_batches:
            break
    end_time = time.time()
    print(f"Time taken for {n_batches} iterations: {end_time - start_time:.2f} seconds")
