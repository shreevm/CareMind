import uvicorn
from pathlib import Path
import sys


repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))


if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8002, reload=False)
