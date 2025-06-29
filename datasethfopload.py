from huggingface_hub import HfApi

api = HfApi()
api.create_repo("ZombitX64/Thai-corpus-iob", repo_type="dataset", exist_ok=True)
api.upload_file(
    path_or_fileobj="data/train_iob_strict.txt",
    path_in_repo="train_iob_strict.txt",
    repo_id="ZombitX64/Thai-corpus-iob",
    repo_type="dataset"
)