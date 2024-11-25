import random

def query_llm() -> str:
    # Respuestas random que simulan una query a Llama

    responses = [
        "Random answer 1",
        "Random answer 2",
        "Random answer 3",
        "Random answer 4"
    ]

    return random.choice(responses)