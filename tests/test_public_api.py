import subprocess
import sys


def test_data_package_import_is_lazy():
    code = """
import sys
import chimera.data
assert 'lightning' not in sys.modules
assert 'matplotlib' not in sys.modules
assert 'torchvision' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_dataset_primitives_are_public():
    from chimera.data import TokenDataset, WindowSampledDataset

    assert TokenDataset.__name__ == "TokenDataset"
    assert WindowSampledDataset.__name__ == "WindowSampledDataset"
