"""The whole incremental-indexing scheme rests on chunk ids being deterministic:
same chunk -> same id (so re-ingesting is an upsert, not a duplicate), edited
chunk -> new id (so the stale one gets deleted)."""

from langchain_core.documents import Document

from ragbot.ids import chunk_id, compute_ids


def make(text="hello", source="a.md", start=0):
    return Document(page_content=text, metadata={"source": source, "start_index": start})


def test_same_chunk_yields_same_id():
    assert chunk_id(make()) == chunk_id(make())


def test_edited_text_yields_new_id():
    assert chunk_id(make(text="hello")) != chunk_id(make(text="hello!"))


def test_position_is_part_of_the_id():
    # identical text at a different offset is a different chunk
    assert chunk_id(make(start=0)) != chunk_id(make(start=800))


def test_source_is_part_of_the_id():
    # identical text in another file must not collide
    assert chunk_id(make(source="a.md")) != chunk_id(make(source="b.md"))


def test_missing_metadata_does_not_raise():
    # PDFs and connectors do not always populate start_index
    assert chunk_id(Document(page_content="x", metadata={}))


def test_compute_ids_preserves_order_and_length():
    chunks = [make(text="one"), make(text="two", start=10), make(text="three", start=20)]
    ids = compute_ids(chunks)
    assert ids == [chunk_id(c) for c in chunks]
    assert len(set(ids)) == 3
