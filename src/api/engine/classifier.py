from typing import Optional

def classify_submission(input_type: Optional[str] = None, response_type: Optional[str] = None) -> str:
    if input_type == "audio":
        return "audio"
    if response_type in ("code", "file"):
        return "code"
    return "text"
