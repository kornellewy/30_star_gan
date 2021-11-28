import os
from shutil import copy2
from pathlib import Path


def create_fakedataset(base_path: str, original_dataset: str) -> str:
    # create dirs
    base_path = Path(base_path)
    anno_path = base_path.joinpath("Anno")
    anno_path.mkdir(parents=True, exist_ok=True)
    eval_path = base_path.joinpath("Eval")
    eval_path.mkdir(parents=True, exist_ok=True)
    img_path = base_path.joinpath("Img", "img_align_celeba", "img_align_celeba")
    img_path.mkdir(parents=True, exist_ok=True)
    # create paths form original dataset
    original_annotations_file_path = Path(original_dataset).joinpath(
        "Anno", "list_attr_celeba.txt"
    )
    # create paths form target dataset
    target_annotations_file_path = Path(anno_path).joinpath("list_attr_celeba.txt")

    # copy files
    copy2(
        str(original_annotations_file_path),
        str(target_annotations_file_path),
    )


def load_images(path:str) -> list:
    images = []
    valid_images = [".jpeg", ".jpg", ".png"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        images.append(os.path.join(path, f))
    return images
