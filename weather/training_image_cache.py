"""Cache only the deterministic Resize prefix, retaining every random transform."""
import hashlib
from PIL import Image
from torchvision import transforms


def cache_training_resizes(dataset):
    if getattr(dataset,"_resized_pair_cache",None) is not None:
        raise ValueError("Training resize cache already exists")
    operations=[]
    for transform in (dataset.transform,dataset.ir_transform):
        if not isinstance(transform,transforms.Compose) or not transform.transforms or not isinstance(transform.transforms[0],transforms.Resize):
            raise ValueError("Training image caching requires the existing deterministic Resize prefix")
        resize=transform.transforms[0]
        if tuple(resize.size)!=(256,256):raise ValueError("Only the JMDA Resize256 training prefix is cached")
        operations.append(resize)
    images=[]
    for row in dataset.paired_samples:
        with Image.open(row[0]) as image:visible=operations[0](image.convert("RGB"))
        with Image.open(row[1]) as image:infrared=operations[1](image.convert("L").convert("RGB"))
        images.append((visible,infrared))
    dataset._resized_pair_cache=tuple(images)
    return {"samples":len(images),"bytes":len(images)*2*3*256*256,
        "cached_prefix":"PIL decoding and existing Resize256 only; subsequent Resize is an exact no-op",
        "random_crop_flip_and_RNG_preserved":True,"class_labels_cached":False,
        "cohort_sha256":hashlib.sha256("\n".join(row[0]+"\t"+row[1] for row in dataset.paired_samples).encode()).hexdigest()}
