"""A model question with a bounded re-prompt: a refused answer is asked again with the reason.

A `Repair` from a `Gated` or `Field` channel re-renders the same template with the refusal
appended as data, so the model sees why and the next answer is a new recorded op.
"""

from string.templatelib import Template

from effective.api import Effect, ask_llm
from effective.channels import Repair, render

REPAIRS = 2
"""Re-prompts after the first answer before the refusal stands."""


def asked[S](name: str, template: Template, output: type[S]) -> Effect[S | Repair]:
    prompt, answer = render(template, output=output), Repair("not asked")
    for _ in range(1 + REPAIRS):
        # lint: totality(total): the answer is `S | Repair` with `S` a type parameter, so after
        # the `Repair` arm the capture is the `S` arm.
        match prompt.resolve((yield from ask_llm(name, prompt.messages, dict))):
            case Repair(reason=reason) as answer:
                refused = t"\nThe last answer was refused: {reason:data}\nAnswer again."
                prompt = render(template + refused, output=output)
            case resolved:
                return resolved
    return answer
