"""Downloads the DTD images (Hugging Face mirror cansa/Describable-Textures-Dataset-DTD) into data/. Resumable."""
from huggingface_hub import snapshot_download

p = snapshot_download(repo_id="cansa/Describable-Textures-Dataset-DTD", repo_type="dataset",
                      local_dir=r"F:\DEDS_extra_checks\test5_textures\data", allow_patterns=["images/*", "README.md"],
                      max_workers=2)
print("downloaded to", p)
