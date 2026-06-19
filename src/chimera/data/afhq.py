"""AFHQ DataModule (StarGAN v2 download + materialized uint8 + device-side cast).

Animal Faces-HQ (cat/dog/wild). Fetched via StarGAN v2's ``download.sh`` Dropbox
zip; see :class:`chimera.data.base.StarGANImageDataModule` for the materialization
rationale and :class:`chimera.data.base.ImageDataModule` for the throughput one.
"""

from chimera.data.base import StarGANImageDataModule


class AFHQDataModule(StarGANImageDataModule):
    url = "https://www.dropbox.com/scl/fi/kpxh6hu04eu28yb8l30wy/afhq.zip?rlkey=usjnva71u164xd4rq6ghlab1u"
    archive_name = "afhq.zip"
    dirname = "afhq"


if __name__ == "__main__":
    import time

    dm = AFHQDataModule(batch_size=256)
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
