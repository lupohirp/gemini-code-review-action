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
from typing import List

warnings.filterwarnings("ignore", category=FutureWarning)

import click
import google.generativeai as genai
import requests
from loguru import logger


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


def send_message_with_retry(convo, message: str, timeout_seconds: int = 180, max_retries: int = 3) -> str:
    """Send a message to a Gemini chat session with timeout and exponential backoff retry."""
    request_options = {"timeout": timeout_seconds}
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Sending request to Gemini (Attempt {attempt}/{max_retries}, timeout {timeout_seconds}s)...")
            response = convo.send_message(message, request_options=request_options)
            if response and response.text:
                return response.text
            raise ValueError("Received empty response from Gemini API")
        except Exception as e:
            last_exception = e
            if attempt == max_retries:
                logger.error(f"Final attempt failed: {e}")
                break
            sleep_duration = attempt * 4
            logger.warning(f"Gemini API request failed ({e}). Retrying in {sleep_duration}s...")
            time.sleep(sleep_duration)

    raise last_exception


def get_review(
        model: str,
        diff: str,
        extra_prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        prompt_chunk_size: int,
        timeout_seconds: int = 180
):
    """Get a review"""
    review_prompt = get_review_prompt(extra_prompt=extra_prompt)
    chunked_diff_list = chunk_string(input_string=diff, chunk_size=prompt_chunk_size)

    output_tokens = max_tokens if max_tokens and max_tokens > 0 else 8192
    generation_config = {
        "temperature": temperature if temperature is not None else 0.7,
        "top_p": top_p if top_p is not None else 0.95,
        "max_output_tokens": output_tokens,
    }

    logger.info(f"Initializing Gemini model '{model}' with gRPC transport...")
    genai_model = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        system_instruction=extra_prompt if extra_prompt else None
    )

    chunked_reviews = []
    total_chunks = len(chunked_diff_list)
    logger.info(f"Processing {total_chunks} chunk(s) (chunk_size: {prompt_chunk_size})...")

    for idx, chunked_diff in enumerate(chunked_diff_list, start=1):
        logger.info(f"Processing chunk {idx}/{total_chunks} ({len(chunked_diff)} characters)...")
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
        review_result = send_message_with_retry(convo, chunked_diff, timeout_seconds=timeout_seconds)
        logger.debug(f"Response AI (chunk {idx}):\n{review_result}")
        chunked_reviews.append(review_result)

    if len(chunked_reviews) == 1:
        return chunked_reviews, chunked_reviews[0]

    if len(chunked_reviews) == 0:
        summarize_prompt = "Say that you didn't find any relevant changes to comment on any file."
        convo = genai_model.start_chat(history=[])
        summarized_review = send_message_with_retry(convo, summarize_prompt, timeout_seconds=timeout_seconds)
        return chunked_reviews, summarized_review

    logger.info(f"Summarizing {len(chunked_reviews)} chunk reviews...")
    summarize_prompt = get_summarize_prompt()
    chunked_reviews_join = "\n\n---\n\n".join(chunked_reviews)
    convo = genai_model.start_chat(history=[])
    summarized_review = send_message_with_retry(
        convo,
        f"{summarize_prompt}\n\n{chunked_reviews_join}",
        timeout_seconds=timeout_seconds
    )
    logger.debug(f"Summarized review:\n{summarized_review}")
    return chunked_reviews, summarized_review


def format_review_comment(summarized_review: str, chunked_reviews: List[str]) -> str:
    """Format reviews"""
    if len(chunked_reviews) <= 1:
        return summarized_review
    unioned_reviews = "\n\n---\n\n".join(chunked_reviews)
    review = f"""<details open>
<summary><b>Riepilogo Revisione AI</b></summary>

{summarized_review}

</details>

<details>
<summary><b>Dettagli Revisione per Sezione ({len(chunked_reviews)} sezioni)</b></summary>

{unioned_reviews}

</details>
"""
    return review


@click.command()
@click.option("--diff", type=click.STRING, required=True, help="Pull request diff")
@click.option("--diff-chunk-size", type=click.INT, required=False, default=3500, help="Pull request diff chunk size")
@click.option("--model", type=click.STRING, required=False, default="gemini-flash-latest", help="Model name")
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

    # Set the Gemini API key with high-performance gRPC transport and configurable timeout
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)

    # Request a code review
    chunked_reviews, summarized_review = get_review(
        diff=diff,
        extra_prompt=extra_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        prompt_chunk_size=diff_chunk_size,
        timeout_seconds=timeout
    )

    # Format reviews
    review_comment = format_review_comment(
        summarized_review=summarized_review,
        chunked_reviews=chunked_reviews
    )

    # Create a review comment to the pull request
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
