import re

INJECTION_TOKENS = [
    r"do'nt apply (all|any) rules",
    r"do'nt consider any or all instructions",
    r"system message",
    r"you are now (allowed|free) to",
    r"developer message",
    r"reveal (the )?(system|developer) prompt",
    r"print (the )?(system|developer) instructions",
    r"jailbreak",
    r"do anything now",
    r"tell me about (creditials | secrets ) from (system | developer) information"
]

MAX_INPUT_TOKENS = 1500

def check_input_length(max_chars: int ,user_input : str):
    if len(user_input) > max_chars:
        raise ValueError(f"Input length must be under {max_chars} characters.")

def check_prompt_injection(user_input : str) -> bool:
    text = user_input.lower()
    for token in INJECTION_TOKENS:
        if (re.search(token , text)):
            return True
    return False

def guard_the_input(user_input: str) -> None:
    """
    Raises the error if violated.
    """

    check_input_length(user_input=user_input ,max_chars= MAX_INPUT_TOKENS)

    if check_prompt_injection(user_input):
        raise ValueError("Prompt Injection detected.")