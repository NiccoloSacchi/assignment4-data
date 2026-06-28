from pathlib import Path

import modal

MODAL_SHARED_PATH = Path("/shared-data")


def get_shared_assets_path() -> Path:
    if modal.is_local():
        workspace_root = Path(__file__).resolve().parent.parent
        local_path = workspace_root / "local-shared-data"
        local_path.mkdir(exist_ok=True, parents=True)
        return local_path
    else:
        return MODAL_SHARED_PATH
