import hashlib
from collections import defaultdict
from PIL import Image

ARMS = ("a0_real", "a1_black", "a2_shuffled")

def build_shuffle_map(items, seed: int = 42) -> dict[str, int]:
    """qID -> index of the item whose image will be used.

    Deterministic. GUARANTEES that no qID receives:
      - its own image, nor
      - an image from the SAME video_id  (frames from the same video look too similar)
    """
    n = len(items)
    
    # Sort indices by video_id to ensure elements with same video_id are contiguous
    indices = list(range(n))
    indices.sort(key=lambda i: items[i].video_id)
    
    # Count occurrences to find a safe shift
    video_counts = defaultdict(int)
    for it in items:
        video_counts[it.video_id] += 1
    
    if not video_counts:
        return {}
        
    max_count = max(video_counts.values())
    shift = max_count
    
    shuffle_map = {}
    for i, idx in enumerate(indices):
        donor_idx = indices[(i + shift) % n]
        if items[idx].video_id == items[donor_idx].video_id:
            raise RuntimeError("Data distribution has a single video dominating > 50%, cannot guarantee distinct video_ids")
        shuffle_map[items[idx].request.qID] = donor_idx
        
    return shuffle_map

def apply_arm(image: Image.Image, arm: str, *, donor: Image.Image | None = None) -> Image.Image:
    """a0_real     -> image as-is
       a1_black    -> Image.new("RGB", image.size, (0,0,0))
       a2_shuffled -> donor (mandatory for this arm)
    """
    if arm == "a0_real":
        return image
    elif arm == "a1_black":
        return Image.new("RGB", image.size, (0, 0, 0))
    elif arm == "a2_shuffled":
        if donor is None:
            raise ValueError("donor missing for a2_shuffled")
        return donor
    else:
        raise ValueError(f"Unknown arm: {arm}")

def tensor_fingerprint(image: Image.Image) -> str:
    """sha256 of the PIL.Image bytes. For gate G2."""
    return hashlib.sha256(image.tobytes()).hexdigest()
