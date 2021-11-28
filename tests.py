import os
from shutil import copy2, rmtree
from pathlib import Path
import unittest

from prepare_dataset import DatasetPreperer
from tests_utils import create_fakedataset, load_images


class TestDatasetPreperer(unittest.TestCase):
    def setUp(self) -> None:
        self.real_dataset_path = "J:/kjn_YT/29_cycle_gan_black_white/CelebA"
        self.fake_dataset_path = "fake_CelebA"
        create_fakedataset(
            base_path=self.fake_dataset_path, original_dataset=self.real_dataset_path
        )
        self.preperer = DatasetPreperer()

    def test_file_copy(self) -> None:
        self.preperer.proces_dataset(self.fake_dataset_path)
        self.assertTrue(
            Path(self.fake_dataset_path)
            .joinpath(
                "Img", "img_align_celeba", "img_align_celeba", "list_attr_celeba.txt"
            )
            .exists()
        )

    def test_create_celeba_n(self) -> None:
        self.preperer.create_celeba_n(
            n_images=10,
            target_dataset_path=self.fake_dataset_path,
            source_dataset_path=self.real_dataset_path,
        )
        self.assertTrue(
            len(
                load_images(
                    str(
                        Path(self.fake_dataset_path).joinpath(
                            "Img", "img_align_celeba", "img_align_celeba"
                        )
                    )
                )
            )
            == 10
        )

    def tearDown(self) -> None:
        rmtree(self.fake_dataset_path, ignore_errors=True)
        rmtree(self.fake_dataset_path, ignore_errors=True)


class TestModel(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
