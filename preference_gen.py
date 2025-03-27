from distilabel.models import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, GroupColumns, FormatTextGenerationDPO, PreferenceToArgilla
from distilabel.steps.tasks import TextGeneration, UltraFeedback

from dotenv import load_dotenv
import os

load_dotenv()

# Define custom steps
class SentenceLengthReward(Step):
    @property
    def inputs(self) -> List[str]:
        return ["generations"]

    @property
    def outputs(self) -> List[str]:
        return ["sentence_length_rewards"]

    def process(self, inputs: StepInput) -> StepOutput:
        for input in inputs:
            generations = input["generations"]
            rewards = []
            for gen in generations:
                sentence_count = len(gen.split("."))
                word_count = len(gen.split())
                reward = 1 if 3 <= sentence_count <= 4 and 30 <= word_count <= 40 else 0
                rewards.append(reward)
            input["sentence_length_rewards"] = rewards
            yield input

class CombineRewards(Step):
    @property
    def inputs(self) -> List[str]:
        return ["ratings", "sentence_length_rewards"]

    @property
    def outputs(self) -> List[str]:
        return ["ratings"]

    def process(self, inputs: StepInput) -> StepOutput:
        for input in inputs:
            ratings = input["ratings"]
            sentence_length_rewards = input["sentence_length_rewards"]
            total_scores = [r + s for r, s in zip(ratings, sentence_length_rewards)]
            input["ratings"] = total_scores
            yield input

# Pipeline definition
with Pipeline(name="tarot_dpo") as pipeline:
    
    load_dataset = LoadDataFromHub(
        repo_id= os.getenv("PROMPT_REPO_ID")
    )

    # Text generation tasks
    generate_responses = [
        TextGeneration(
            name="text_generation_sft_awq",
            llm=InferenceEndpointsLLM(
                model_id="sft_awq",
                tokenizer_id="sft_awq",
                generation_kwargs={"max_new_tokens": 512, "temperature": 0.7},
            )
        ),
        TextGeneration(
            name="text_generation_sft_qwen",
            llm=InferenceEndpointsLLM(
                model_id="sft_qwen",
                tokenizer_id="sft_qwen",
                generation_kwargs={"max_new_tokens": 512, "temperature": 0.7},
            )
        ),
    ]

    # Group responses
    group_responses = GroupColumns(
        name="group_responses",
        columns=["generation", "model_name"],
        output_columns=["generations", "model_names"]
    )

    # Custom sentence length reward step
    sentence_length_reward = SentenceLengthReward(name="sentence_length_reward")

    # UltraFeedback for quality scoring (assumed part of your pipeline)
    ultrafeedback = UltraFeedback(
        name="ultrafeedback",
        llm=InferenceEndpointsLLM(
            model_id="Qwen/QwQ-32B",  # Replace with your evaluator LLM
            generation_kwargs={"max_new_tokens": 512}
        )
    )

    # Combine rewards
    combine_rewards = CombineRewards(name="combine_rewards")

    # Format for DPO using total scores
    format_dpo = FormatTextGenerationDPO()

    # Connect steps
    for task in generate_responses:
        load_dataset >> task >> group_responses

    group_responses >> sentence_length_reward >> combine_rewards
    group_responses >> ultrafeedback >> combine_rewards
    combine_rewards >> format_dpo

# Run the pipeline
if __name__ == "__main__":
    distiset = pipeline.run()
    # Optionally push to Hugging Face Hub
    distiset.push_to_hub(os.getenv("PREF_REPO_ID"))