from app.mmr_utils import cosine_sim, count_redundant_items


def test_count_redundant_items_detects_duplicates():
    # Two identical vectors and one different vector
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]   # duplicate of v1
    v3 = [0.0, 1.0, 0.0]   # distinct

    items = [
        {"chunk_id": "a"},
        {"chunk_id": "b"},
        {"chunk_id": "c"},
    ]
    vectors = {"a": v1, "b": v2, "c": v3}

    # with threshold < 1.0, b is redundant with a
    redundant = count_redundant_items(items, vectors, threshold=0.95)
    assert redundant == 1


def test_cosine_sim_basic():
    assert abs(cosine_sim([1, 0], [1, 0]) - 1.0) < 1e-6
    assert abs(cosine_sim([1, 0], [0, 1]) - 0.0) < 1e-6
