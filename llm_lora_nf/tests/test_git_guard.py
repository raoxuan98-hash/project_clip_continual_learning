from llm_lora_nf.git_guard import MAX_FILE_BYTES, PayloadEntry, validate_payload


def test_payload_guard_accepts_model_code():
    assert validate_payload(
        [PayloadEntry("llm_lora_nf/src/llm_lora_nf/layers.py", 2048)]
    ) == []


def test_payload_guard_rejects_history_and_large_files():
    violations = validate_payload(
        [
            PayloadEntry("chat-history/secret.md", 10),
            PayloadEntry("llm_lora_nf/model.safetensors", MAX_FILE_BYTES + 1),
        ]
    )
    assert any("forbidden path" in item for item in violations)
    assert any("exceeds 10 MiB" in item for item in violations)

