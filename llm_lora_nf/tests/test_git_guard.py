from llm_lora_nf.git_guard import MAX_FILE_BYTES, PayloadEntry, validate_payload


def test_payload_guard_accepts_model_code():
    assert validate_payload(
        [PayloadEntry("llm_lora_nf/src/llm_lora_nf/layers.py", 2048)]
    ) == []


def test_payload_guard_accepts_only_declared_supporting_assets():
    assert validate_payload(
        [
            PayloadEntry(".gitignore", 2048),
            PayloadEntry("docs/llm-lora-nf/GOAL.md", 2048),
            PayloadEntry("llm_lora_nf/requirements-server.txt", 2048),
            PayloadEntry("llm_lora_nf/configs/paper/math_base.yaml", 2048),
        ]
    ) == []


def test_payload_guard_rejects_unknown_or_unsafe_paths():
    violations = validate_payload(
        [
            PayloadEntry("notes/private.txt", 8),
            PayloadEntry("../outside.py", 8),
        ]
    )
    assert any("outside allowed LLM code payload" in item for item in violations)
    assert any("unsafe payload path" in item for item in violations)


def test_payload_guard_rejects_history_and_large_files():
    violations = validate_payload(
        [
            PayloadEntry("chat-history/secret.md", 10),
            PayloadEntry("llm_lora_nf/model.safetensors", MAX_FILE_BYTES + 1),
        ]
    )
    assert any("forbidden path" in item for item in violations)
    assert any("forbidden artifact extension" in item for item in violations)
    assert any("exceeds 10 MiB" in item for item in violations)


def test_payload_guard_rejects_calibration_cache_and_small_weight_file():
    violations = validate_payload(
        [
            PayloadEntry(
                "llm_lora_nf/calibration_cache/key/moments.pt",
                8,
            ),
            PayloadEntry("llm_lora_nf/tiny-model.bin", 8),
            PayloadEntry(
                "llm_lora_nf/request_cache/key/requests.pickle",
                8,
            ),
        ]
    )
    assert sum("forbidden artifact extension" in item for item in violations) == 3
    assert sum("forbidden path" in item for item in violations) == 2
