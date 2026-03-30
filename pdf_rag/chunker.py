def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Recursively split text into chunks of approximately chunk_size characters with overlap."""
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, separators: list[str]) -> list[str]:
        if not text:
            return []
        if len(text) <= chunk_size:
            return [text]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            # Hard split by character
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

        parts = text.split(sep)
        chunks = []
        current = ""

        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    if len(current) > chunk_size and remaining_seps:
                        chunks.extend(_split(current, remaining_seps))
                    else:
                        chunks.append(current)
                current = part

        if current:
            if len(current) > chunk_size and remaining_seps:
                chunks.extend(_split(current, remaining_seps))
            else:
                chunks.append(current)

        return chunks

    raw_chunks = _split(text, separators)

    # Apply overlap: prepend tail of previous chunk to each chunk
    result = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0 or not result:
            result.append(chunk)
        else:
            tail = result[-1][-overlap:] if len(result[-1]) > overlap else result[-1]
            result.append(tail + " " + chunk)

    return [c.strip() for c in result if c.strip()]
