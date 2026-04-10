from api.engine.classifier import classify_submission

def test_text_input_type():
    assert classify_submission(input_type="text", response_type="chat") == "text"

def test_audio_input_type():
    assert classify_submission(input_type="audio", response_type="chat") == "audio"

def test_code_response_type():
    assert classify_submission(input_type="text", response_type="code") == "code"

def test_file_response_type():
    assert classify_submission(input_type="text", response_type="file") == "code"

def test_defaults_to_text():
    assert classify_submission(input_type=None, response_type=None) == "text"
