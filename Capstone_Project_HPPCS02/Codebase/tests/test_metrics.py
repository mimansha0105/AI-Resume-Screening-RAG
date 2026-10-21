from rag_system.evaluate import recall_at_k, precision_at_k, reciprocal_rank
from rag_system.pipeline import rewrite_query, semantic_chunks

def test_recall_at_k():
    assert recall_at_k(["cv1", "cv2", "cv3"], {"cv2", "cv4"}, 3) == 0.5

def test_precision_at_k():
    assert precision_at_k(["cv1", "cv2", "cv3"], {"cv2"}, 3) == 1 / 3

def test_reciprocal_rank():
    assert reciprocal_rank(["cv1", "cv2", "cv3"], {"cv3"}) == 1 / 3

def test_query_rewriting():
    assert "kubernetes" in rewrite_query("Need k8s experience").lower()

def test_semantic_chunking():
    chunks = semantic_chunks("Skills\nPython, AWS\nExperience\nBuilt APIs")
    assert any(c["section"] == "skills" for c in chunks)
    assert any(c["section"] == "experience" for c in chunks)
