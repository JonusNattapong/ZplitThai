from huggingface_hub import HfApi

api = HfApi()
api.create_repo("ZombitX64/Thaitokenizer", repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="AdvancedThaiTokenizerV3",
    repo_id="ZombitX64/Thaitokenizer",
    repo_type="model"
)