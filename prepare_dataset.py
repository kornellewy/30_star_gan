import os
from shutil import copy2
from pathlib import Path
from typing import Tuple


class DatasetPreperer:
    def __init__(self) -> None:
        super().__init__()
        self.annotations_file_name = "list_attr_celeba.txt"
        self.start_annotations_file_location = ""
        self.end_annotations_file_location = ""

    def proces_dataset(self, dataset_path: str) -> None:
        return self._proces_dataset(dataset_path)

    def _proces_dataset(self, dataset_path: str) -> None:
        self._copy_annotations_file(dataset_path)

    def _copy_annotations_file(self, dataset_path: str) -> None:
        self.start_annotations_file_location = Path(dataset_path).joinpath(
            "Anno", self.annotations_file_name
        )
        self.end_annotations_file_location = Path(dataset_path).joinpath(
            "Img", "img_align_celeba", "img_align_celeba", self.annotations_file_name
        )
        copy2(self.start_annotations_file_location, self.end_annotations_file_location)

    def create_celeba_n(
        self, n_images: int, target_dataset_path: str, source_dataset_path: str
    ):
        (
            anno_dir_path,
            eval_dir_path,
            imgs_dir_path,
        ) = self._create_target_dataset_structure(base_dir_path=target_dataset_path)
        annotations_file_location = self._copy_modify_annotations_file_with_n_images(
            n_images=n_images,
            target_imgs_dir_path=imgs_dir_path,
            source_dataset_path=source_dataset_path,
        )
        self._copy_n_images(
            annotations_file_location=annotations_file_location,
            target_imgs_dir_path=imgs_dir_path,
            source_dataset_path=source_dataset_path,
        )

    def _create_target_dataset_structure(
        self, base_dir_path: str
    ) -> Tuple[Path, Path, Path]:
        base_dir_path = Path(base_dir_path)
        anno_dir_path = base_dir_path.joinpath("Anno")
        anno_dir_path.mkdir(parents=True, exist_ok=True)
        eval_dir_path = base_dir_path.joinpath("Eval")
        eval_dir_path.mkdir(parents=True, exist_ok=True)
        imgs_dir_path = base_dir_path.joinpath(
            "Img", "img_align_celeba", "img_align_celeba"
        )
        imgs_dir_path.mkdir(parents=True, exist_ok=True)
        return anno_dir_path, eval_dir_path, imgs_dir_path

    def _copy_modify_annotations_file_with_n_images(
        self,
        n_images: int,
        target_imgs_dir_path: str,
        source_dataset_path: str,
    ) -> str:
        start_annotations_file_location = Path(source_dataset_path).joinpath(
            "Img", "img_align_celeba", "img_align_celeba", self.annotations_file_name
        )
        end_annotations_file_location = Path(target_imgs_dir_path).joinpath(
            self.annotations_file_name
        )
        lines = [line.rstrip() for line in open(start_annotations_file_location, "r")][
            : n_images + 1
        ]
        self.saveListToFile(
            list_of_names=lines, path_to_file=end_annotations_file_location
        )
        return end_annotations_file_location

    def _copy_n_images(
        self,
        annotations_file_location: str,
        target_imgs_dir_path: str,
        source_dataset_path: str,
    ) -> None:
        source_imgs_dir_path = Path(source_dataset_path).joinpath(
            "Img", "img_align_celeba", "img_align_celeba"
        )
        # read all images paths form file
        lines = [line.rstrip() for line in open(annotations_file_location, "r")]
        images_names = lines[0].split()
        for img_idx, line in enumerate(lines[1:]):
            filename, *values = line.split()
            source_img_path = str(Path(source_imgs_dir_path).joinpath(filename))
            target_img_path = str(Path(target_imgs_dir_path).joinpath(filename))
            copy2(source_img_path, target_img_path)

    @staticmethod
    def saveListToFile(list_of_names: list, path_to_file: str) -> None:
        file1 = open(path_to_file, "w")
        for name in list_of_names:
            file1.writelines(f"{name}\n")
        file1.close()


if __name__ == "__main__":
    # prepere original dataset
    celeba_dataset_path = "J:/kjn_YT/29_cycle_gan_black_white/CelebA"
    preperer = DatasetPreperer()
    preperer.proces_dataset(dataset_path=celeba_dataset_path)
    # create small 10000 celeba
    new_dataset_path = "CelebA_10000"
    preperer.create_celeba_n(
        n_images=10000,
        target_dataset_path=new_dataset_path,
        source_dataset_path=celeba_dataset_path,
    )
