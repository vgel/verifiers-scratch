import os

from datasets import load_dataset
from openai import OpenAI

import verifiers as vf


def load_environment(
    dataset_name: str = "agentlans/wikipedia-paragraphs",
    dataset_split: str | None = "train",
    dataset_key: str = "text",
    judge_model: str = "gpt-4.1-mini",
    judge_base_url: str = "https://api.openai.com/v1",
    judge_api_key_var: str = "OPENAI_API_KEY",
) -> vf.Environment:
    dataset = load_dataset(dataset_name, split=dataset_split)
    dataset = dataset.map(
        lambda x: {"question": x[dataset_key], "answer": x[dataset_key]}
    )

    judge_client = OpenAI(api_key=os.getenv(judge_api_key_var), base_url=judge_base_url)

    # Create a comprehensive evaluation prompt
    judge_prompt = """Evaluate this base model contination from a prefix, compared to the true continuation from Wikipedia.

<prefix>
{prefix}
</prefix>

<true_continuation>
{true_continuation}
</prefix>

<model_continuation>
{model_continuation}
</model_continuation>

Provide a letter grade from A-F where:
- A: Smooth prose, facts are accurate w.r.t the true continuation, uses Wikipedia formatting
- B-C: Awkward writing, incorrect facts / dates / names, incorrect formatting
- D-F: Incoherent text, sampling errors, repetition / looping

Respond with the letter grade in <grade> ... </grade> tags."""

    judge_parser = vf.XMLParser(fields=[], answer_field="grade")

    rubric = vf.JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=judge_prompt,
        parser=judge_parser,
    )

    def grade_reward(prompt, completion, answer, state, **kwargs) -> float:
        judge_response = rubric.judge(prompt, completion, answer, state, **kwargs)
        judge_grade = (
            (judge_parser.parse_answer(judge_response) or "F")
            .strip()
            .replace("+", "")
            .replace("-", "")
            .upper()
        )
        return {
            "A": 1.0,
            "B": 0.75,
            "C": 0.5,
            "D": 0.25,
        }.get(judge_grade, 0.0)

    rubric.add_reward_func(grade_reward, weight=1.0)

    return vf.SingleTurnEnv(
        message_type="completion",
        dataset=dataset,
        parser=vf.Parser(),
        rubric=rubric,
    )