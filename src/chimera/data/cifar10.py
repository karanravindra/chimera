"""CIFAR-10 DataModule (materialized uint8 + device-side cast).

See :class:`chimera.data.base.ImageDataModule` for the throughput rationale.
"""

from torchvision import datasets

from chimera.data.base import ImageDataModule


class CIFAR10DataModule(ImageDataModule):
    dataset_cls = datasets.CIFAR10


if __name__ == "__main__":
    import time

    dm = CIFAR10DataModule(batch_size=256)
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
