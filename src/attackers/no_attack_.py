"""Prompt-conditioned erased-model baseline."""


def generate(task, prompt, seed, guidance):
    return task.sample(prompt, seed, guidance)
