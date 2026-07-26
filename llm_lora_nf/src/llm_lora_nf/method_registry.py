TRACK_B_NATIVE_METHODS = frozenset(
    {"lora_nf", "lora_null", "milora"}
)
TRACK_B_DIRECT_PEFT_METHODS = frozenset(
    {"lora", "dora", "pissa"}
)
TRACK_B_PEFT_METHODS = frozenset(
    {*TRACK_B_DIRECT_PEFT_METHODS, "corda"}
)
TRACK_B_METHODS = frozenset(
    {*TRACK_B_NATIVE_METHODS, *TRACK_B_PEFT_METHODS}
)
