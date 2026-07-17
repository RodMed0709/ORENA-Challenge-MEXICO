def extract_items(text: str) -> int:
    if not text:
        return 0
        
    # The prefill "ITEMS:" is not returned by generation.
    # The first line of the output contains the comma-separated items.
    first_line = text.split("\n")[0].strip()
    if not first_line:
        return 0
        
    return len([x for x in first_line.split(",") if x.strip()])
