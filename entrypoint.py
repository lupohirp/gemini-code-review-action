#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import json
import os
import time
import warnings
from typing import List, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)

import click
import google.generativeai as genai
import requests
from loguru import logger

FALLBACK_CHAIN = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite",
]


def check_required_env_vars():
    """Check required environment variables"""
    required_env_vars = [
        "GEMINI_API_KEY",
        "GITHUB_TOKEN",
        "GITHUB_REPOSITORY",
        "GITHUB_PULL_REQUEST_NUMBER",
        "GIT_COMMIT_HASH",
    ]
    for required_env_var in required_env_vars:
        if os.getenv(required_env_var) is None:
            raise ValueError(f"{required_env_var} is not set")


def get_review_prompt(extra_prompt: str = "") -> str:
    """Get a prompt template"""
    template = """
    This is a pull request or part of a pull request if the pull request is very large.
    Suppose you review this PR as an excellent software engineer and an excellent security engineer.
    Can you tell me the issues with differences in a pull request and provide suggestions to improve it?
    You can provide a review summary and issue comments per file if any major issues are found.
    Always include the name of the file that is citing the improvement or problem.
    In the next messages I will be sending you the difference between the GitHub file codes, okay?
    """
    return template


def get_summarize_prompt() -> str:
    """Get a prompt template"""
    template = """
    Can you summarize this for me?
    It would be good to stick to highlighting pressing issues and providing code suggestions to improve the pull request.
    Here's what you need to summarize:
    """
    return template


def create_a_comment_to_pull_request(
        github_token: str,
        github_repository: str,
        pull_request_number: int,
        git_commit_hash: str,
        body: str):
    """Create a comment to a pull request"""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"Bearer {github_token}"
    }
    data = {
        "body": body,
        "commit_id": git_commit_hash,
        "event": "COMMENT"
    }
    url = f"https://api.github.com/repos/{github_repository}/pulls/{pull_request_number}/reviews"
    response = requests.post(url, headers=headers, json=data, timeout=30)
    if not response.ok:
        logger.error(f"GitHub review API failed (HTTP {response.status_code}): {response.text}")
        response.raise_for_status()
    logger.info(f"Successfully posted code review to PR #{pull_request_number}")
    return response


def chunk_string(input_string: str, chunk_size: int) -> List[str]:
    """Chunk a string"""
    if not input_string:
        return []
    chunked_inputs = []
    for i in range(0, len(input_string), chunk_size):
        chunked_inputs.append(input_string[i:i + chunk_size])
    return chunked_inputs


def is_rate_limit_error(exc: Exception) -> bool:
    """Check if exception is caused by 429 quota or rate limits."""
    msg = str(exc).lower()
    return (
        "429" in msg
        or "quota" in msg
        or "resourceexhausted" in msg
        or "rate limit" in msg
        or "exceeded" in msg
    )


def send_message_with_timeout(convo, message: str, timeout_seconds: int = 180) -> str:
    """Send a message to a Gemini chat session with timeout."""
    request_options = {"timeout": timeout_seconds}
    response = convo.send_message(message, request_options=request_options)
    if response and response.text:
        return response.text
    raise ValueError("Received empty response from Gemini API")


def get_review_for_model(
        model: str,
        diff: str,
        extra_prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        prompt_chunk_size: int,
        timeout_seconds: int = 180
) -> Tuple[List[str], str]:
    """Execute review for a specific model."""
    review_prompt = get_review_prompt(extra_prompt=extra_prompt)
    chunked_diff_list = chunk_string(input_string=diff, chunk_size=prompt_chunk_size)

    output_tokens = max_tokens if max_tokens and max_tokens > 0 else 8192
    generation_config = {
        "temperature": temperature if temperature is not None else 0.7,
        "top_p": top_p if top_p is not None else 0.95,
        "max_output_tokens": output_tokens,
    }

    genai_model = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        system_instruction=extra_prompt if extra_prompt else None
    )

    chunked_reviews = []
    total_chunks = len(chunked_diff_list)

    for idx, chunked_diff in enumerate(chunked_diff_list, start=1):
        if idx > 1:
            time.sleep(3)

        logger.info(f"[{model}] Processing chunk {idx}/{total_chunks} ({len(chunked_diff)} chars)...")
        convo = genai_model.start_chat(history=[
            {
                "role": "user",
                "parts": [review_prompt]
            },
            {
                "role": "model",
                "parts": ["Ok"]
            },
        ])
        review_result = send_message_with_timeout(convo, chunked_diff, timeout_seconds=timeout_seconds)
        chunked_reviews.append(review_result)

    if len(chunked_reviews) == 1:
        return chunked_reviews, chunked_reviews[0]

    if len(chunked_reviews) == 0:
        summarize_prompt = "Say that you didn't find any relevant changes to comment on any file."
        convo = genai_model.start_chat(history=[])
        summarized_review = send_message_with_timeout(convo, summarize_prompt, timeout_seconds=timeout_seconds)
        return chunked_reviews, summarized_review

    logger.info(f"[{model}] Summarizing {len(chunked_reviews)} chunk reviews...")
    summarize_prompt = get_summarize_prompt()
    chunked_reviews_join = "\n\n---\n\n".join(chunked_reviews)
    convo = genai_model.start_chat(history=[])
    summarized_review = send_message_with_timeout(
        convo,
        f"{summarize_prompt}\n\n{chunked_reviews_join}",
        timeout_seconds=timeout_seconds
    )
    return chunked_reviews, summarized_review


def execute_review_with_fallback(
        requested_model: str,
        diff: str,
        extra_prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        prompt_chunk_size: int,
        timeout_seconds: int = 180
) -> Tuple[List[str], str, str]:
    """Iterate through candidate models chain: 3.8 -> 3.7 -> 3.5 flash-lite on RPD/quota limit."""
    candidate_models = []
    if requested_model and requested_model not in FALLBACK_CHAIN:
        candidate_models.append(requested_model)
    for m in FALLBACK_CHAIN:
        if m not in candidate_models:
            candidate_models.append(m)

    logger.info(f"Candidate models rotation chain: {candidate_models}")

    last_error = None
    for model_name in candidate_models:
        try:
            logger.info(f"===> Attempting review with model '{model_name}'...")
            chunked_reviews, summarized_review = get_review_for_model(
                model=model_name,
                diff=diff,
                extra_prompt=extra_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                prompt_chunk_size=prompt_chunk_size,
                timeout_seconds=timeout_seconds
            )
            logger.info(f"Review successfully completed with model '{model_name}'!")
            return chunked_reviews, summarized_review, model_name
        except Exception as e:
            last_error = e
            if is_rate_limit_error(e):
                logger.warning(
                    f"Quota / Rate limit reached on model '{model_name}' ({e}). "
                    f"Failing over immediately to next model in chain..."
                )
            else:
                logger.warning(
                    f"Model '{model_name}' failed ({e}). "
                    f"Failing over to next model in chain..."
                )

    raise RuntimeError(f"All candidate models failed in fallback chain. Last error: {last_error}")


def format_review_comment(summarized_review: str, chunked_reviews: List[str], model_name: str) -> str:
    """Format reviews with model badge"""
    model_badge = f"\n\n---\n*Revisione del codice eseguita con Google Gemini (`{model_name}`)*"
    if len(chunked_reviews) <= 1:
        return summarized_review + model_badge
    unioned_reviews = "\n\n---\n\n".join(chunked_reviews)
    review = f"""<details open>
<summary><b>Riepilogo Revisione AI</b></summary>

{summarized_review}

</details>

<details>
<summary><b>Dettagli Revisione per Sezione ({len(chunked_reviews)} sezioni)</b></summary>

{unioned_reviews}

</details>
{model_badge}
"""
    return review


@click.command()
@click.option("--diff", type=click.STRING, required=True, help="Pull request diff")
@click.option("--diff-chunk-size", type=click.INT, required=False, default=65000, help="Pull request diff chunk size")
@click.option("--model", type=click.STRING, required=False, default="gemini-3.8-flash", help="Initial model name")
@click.option("--extra-prompt", type=click.STRING, required=False, default="", help="Extra prompt")
@click.option("--temperature", type=click.FLOAT, required=False, default=0.7, help="Temperature")
@click.option("--max-tokens", type=click.INT, required=False, default=8192, help="Max tokens")
@click.option("--top-p", type=click.FLOAT, required=False, default=0.95, help="Top P")
@click.option("--frequency-penalty", type=click.FLOAT, required=False, default=0.0, help="Frequency penalty")
@click.option("--presence-penalty", type=click.FLOAT, required=False, default=0.0, help="Presence penalty")
@click.option("--timeout", type=click.INT, required=False, default=180, help="Timeout in seconds per Gemini request")
@click.option("--log-level", type=click.STRING, required=False, default="INFO", help="Log level")
def main(
        diff: str,
        diff_chunk_size: int,
        model: str,
        extra_prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        timeout: int,
        log_level: str
):
    # Set log level
    logger.level(log_level)
    # Check if necessary environment variables are set
    check_required_env_vars()

    # Configure Gemini with high-performance gRPC transport
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)

    # Request code review with immediate model failover on quota / RPD limits (3.8 -> 3.7 -> 3.5 flash-lite)
    chunked_reviews, summarized_review, successful_model = execute_review_with_fallback(
        requested_model=model,
        diff=diff,
        extra_prompt=extra_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        prompt_chunk_size=diff_chunk_size,
        timeout_seconds=timeout
    )

    # Format review comment
    review_comment = format_review_comment(
        summarized_review=summarized_review,
        chunked_reviews=chunked_reviews,
        model_name=successful_model
    )

    # Post review comment to GitHub PR
    create_a_comment_to_pull_request(
        github_token=os.getenv("GITHUB_TOKEN"),
        github_repository=os.getenv("GITHUB_REPOSITORY"),
        pull_request_number=int(os.getenv("GITHUB_PULL_REQUEST_NUMBER")),
        git_commit_hash=os.getenv("GIT_COMMIT_HASH"),
        body=review_comment
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
