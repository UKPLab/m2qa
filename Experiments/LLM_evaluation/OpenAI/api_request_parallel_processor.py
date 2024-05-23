# Adapted from: https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import re  # for matching endpoint from request URL
import time  # for sleeping after rate limit is hit
from dataclasses import (  # for storing API inputs, outputs, and metadata
    dataclass,
    field,
)

# imports
import aiohttp  # for making API calls concurrently
import tiktoken  # for counting tokens


async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.2  # sleep so that not all requests are send at once

    # initialize logging
    logging.basicConfig(level=logging.INFO)

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}
    # use api-key header for Azure deployments
    if "/deployments" in request_url:
        request_header = {"api-key": f"{api_key}"}

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    status_tracker = StatusTracker()  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    # initialize file reading
    with open(requests_filepath) as file:
        # `requests` will provide requests one at a time
        loaded_list = json.load(file)

        # Iterator (generator) for the requests
        requests = iter(loaded_list)

        logging.debug("File opened. Entering main loop")
        async with aiohttp.ClientSession() as session:  # Initialize ClientSession here
            while True:
                # get next request (if one is not already waiting for capacity)
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = queue_of_requests_to_retry.get_nowait()
                        logging.debug(f"Retrying request {next_request.example_id}: {next_request}")
                    elif file_not_finished:
                        try:
                            # get new request
                            request_json = next(requests)

                            # request_json includes attributes for ...
                            # ... the requests (temperature, max_tokens, model)
                            # ... the task (example_id)

                            example_id = request_json.pop("example_id")
                            prompt_used = request_json.pop("prompt_used")

                            next_request = APIRequest(
                                example_id=example_id,
                                prompt_used=prompt_used,
                                request_json=request_json,
                                token_consumption=num_tokens_consumed_from_request(
                                    request_json, api_endpoint, token_encoding_name
                                ),
                                attempts_left=max_attempts,
                                # don't need it. Only thing I need is an additional parameter to store what prompt was used (name of the prompt)
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logging.debug(f"Reading request {next_request.example_id}: {next_request}")
                        except StopIteration:
                            # if file runs out, set flag to stop reading it
                            logging.debug("Read file exhausted")
                            file_not_finished = False

                # update available capacity
                current_time = time.time()
                seconds_since_update = current_time - last_update_time
                available_request_capacity = min(
                    available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
                    max_requests_per_minute,
                )
                available_token_capacity = min(
                    available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
                    max_tokens_per_minute,
                )
                last_update_time = current_time

                # if enough capacity available, call API
                if next_request:
                    next_request_tokens = next_request.token_consumption
                    if available_request_capacity >= 1 and available_token_capacity >= next_request_tokens:
                        # update counters
                        available_request_capacity -= 1
                        available_token_capacity -= next_request_tokens
                        next_request.attempts_left -= 1

                        # call API
                        asyncio.create_task(
                            next_request.call_api(
                                session=session,
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=queue_of_requests_to_retry,
                                save_filepath=save_filepath,
                                status_tracker=status_tracker,
                            )
                        )
                        next_request = None  # reset next_request to empty

                # if all tasks are finished, break
                if status_tracker.num_tasks_in_progress == 0:
                    break

                # main loop sleeps briefly so concurrent tasks can run
                await asyncio.sleep(seconds_to_sleep_each_loop)

                # if a rate limit error was hit recently, pause to cool down
                seconds_since_rate_limit_error = time.time() - status_tracker.time_of_last_rate_limit_error
                if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
                    remaining_seconds_to_pause = (
                        seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error
                    )
                    await asyncio.sleep(remaining_seconds_to_pause)
                    # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                    logging.warn(
                        f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
                    )

        # after finishing, log final status
        logging.info(f"""Parallel processing complete. Results saved to {save_filepath}""")
        if status_tracker.num_tasks_failed > 0:
            logging.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}."
            )
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
            )


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other data. Contains a method to make an API call."""

    example_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    prompt_used: str
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{self.example_id}")
        error = None
        try:
            async with session.post(url=request_url, headers=request_header, json=self.request_json) as response:
                response = await response.json()
            if "error" in response:
                logging.warning(f"Request {self.example_id} failed with error {response['error']}")
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1  # rate limit errors are counted separately

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.example_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(
                    f"Request {self.request_json} failed after all attempts. Tried to send to endpoint: {request_url}\nSaving errors: {self.result}\n"
                )

                data = {
                    "example_id": self.example_id,
                    "prompt_used": self.prompt_used,
                    "request_json": self.request_json,
                }

                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            print(f"response: {response}")
            predicted_answer = response["choices"][0]["message"]["content"]

            if len(response["choices"]) > 1:
                print(f"Has multiple response choices: {response['choices']}")

            if (
                self.prompt_used == "five_shot_cross_domain"
                or self.prompt_used == "five_shot_cross_lingual"
                or self.prompt_used == "zero_shot_same_language"
            ):
                if (
                    predicted_answer == "unbeantwortbar"
                    or predicted_answer == "cevaplanamaz"
                    or predicted_answer == "无法回答"
                ):
                    data = {
                        "prediction_text": "unanswerable",
                        "id": self.example_id,
                        "no_answer_probability": 1.0,
                    }
                else:
                    data = {
                        "prediction_text": predicted_answer,
                        "id": self.example_id,
                        "no_answer_probability": 0.0,
                    }
            else:
                # English Prompt
                if predicted_answer == "unanswerable":
                    data = {
                        "prediction_text": "unanswerable",
                        "id": self.example_id,
                        "no_answer_probability": 1.0,
                    }
                else:
                    data = {
                        "prediction_text": predicted_answer,
                        "id": self.example_id,
                        "no_answer_probability": 0.0,
                    }
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.example_id} saved to {save_filepath}")


def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""
    match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
    if match is None:
        # for Azure OpenAI deployment urls
        match = re.search(r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url)
    return match[1]


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data, ensure_ascii=False)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)

    # Chat completions
    max_tokens = request_json.get("max_tokens", 15)
    n = request_json.get("n", 1)
    completion_tokens = n * max_tokens

    # chat completions
    num_tokens = 0
    for message in request_json["messages"]:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens -= 1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens + completion_tokens


def run_openai_requests(
    requests_filepath,
    save_filepath,
    request_url,
    api_key,
    max_requests_per_minute,
    max_tokens_per_minute,
    token_encoding_name="cl100k_base",
    max_attempts=15,
):
    if (
        requests_filepath is None
        or save_filepath is None
        or request_url is None
        or api_key is None
        or max_requests_per_minute is None
        or max_tokens_per_minute is None
    ):
        raise ValueError("All arguments must be provided.")

    # run script
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=requests_filepath,
            save_filepath=save_filepath,
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name=token_encoding_name,
            max_attempts=max_attempts,
        )
    )
