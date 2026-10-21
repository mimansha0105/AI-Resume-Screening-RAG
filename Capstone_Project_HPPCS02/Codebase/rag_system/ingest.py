from pathlib import Path
from .pipeline import ResumeRAG

if __name__ == "__main__":
    codebase = Path(__file__).resolve().parents[1]
    print(ResumeRAG().ingest_directory(str(codebase)))
