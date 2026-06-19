"""MNIST DataModule (materialized uint8 + device-side cast).

See :class:`chimera.data.base.ImageDataModule` for the throughput rationale.
``gpu_transform`` lets callers swap the default cast for batched on-GPU
augmentation (e.g. DINO multi-view).
"""

from torchvision import datasets

from chimera.data.base import ImageDataModule


class MNISTDataModule(ImageDataModule):
    dataset_cls = datasets.MNIST


if __name__ == "__main__":
    import time

    dm = MNISTDataModule(batch_size=256)
    dm.prepare_data()
    dm.setup("fit")
    train_loader = dm.train_dataloader()

    # evaluate iteration speed (raw uint8 batches; cast happens on-device)
    n_batches = 100
    start_time = time.time()
    for i, batch in enumerate(train_loader):
        if i + 1 >= n_batches:
            break
    end_time = time.time()
    print(f"Time taken for {n_batches} iterations: {end_time - start_time:.2f} seconds")
