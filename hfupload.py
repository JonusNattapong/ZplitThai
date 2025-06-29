from huggingface_hub import HfApi

api = HfApi()
api.create_repo("ZombitX64/bitthaitokenizer", repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="Bitthaitokenizer",
    repo_id="ZombitX64/bitthaitokenizer",
    repo_type="model"
)