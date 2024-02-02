"""
WIP: SageMaker Module for using any LM behind SageMaker endpoints
"""
from typing import Any, Dict, Literal, Optional, Union

import aws_assume_role_lib  # Only available in PyPI
import boto3
import sagemaker
from botocore.client import Config


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details)
    )


class SageMaker(LM):
    """Wrapper around AWS Bedrock API.

    Args:
        model (str, optional): OpenAI or Azure supported LLM model to use. Defaults to "text-davinci-002".
        api_key (Optional[str], optional): API provider Authentication token. use Defaults to None.
        api_provider (Literal["openai", "azure"], optional): The API provider to use. Defaults to "openai".
        model_type (Literal["chat", "text"], optional): The type of model that was specified. Mainly to decide the optimal prompting strategy. Defaults to "text".
        **kwargs: Additional arguments to pass to the API provider.
    """

    DEFAULT_GENERATION_CONFIG = {
        "max_tokens_to_sample": 100,
        "temperature" :0,
        "top_k": 1,
        "stop_sequences": ["\n\nHuman:"],
    }

    def __init__(
        self,
        endpoint_name: str,
        region: str,
        invocation_role_arn: str,
        model_config: Optional[Dict] = None,
        model_type: Literal["chat", "text"] = "text",
        **kwargs,
    ):
        super().__init__(model)
        self.provider = "bedrock"
        
        self.endpoint_name = endpoint_name
        self.region = region
        self.model_type = model_type
        self.model_config = (
            DEFAULT_GENERATION_CONFIG if not model_config else
            {**DEFAULT_GENERATION_CONFIG, **model_config}
        )

        # For caching
        self.kwargs["endpoint_name"] = self.endpoint_name
        self.kwargs["region"] = self.region
        self.kwargs["model_config"] = self.model_config
        # self.kwargs["_content_type"] = "application/json"

        self._boto_session = boto3.Session()
        self._assumed_role_session = aws_assume_role_lib.assume_role(
            session=self.boto_session,
            RoleArn=invocation_role_arn,
            SourceIdentity=aws_assume_role_lib.generate_lambda_session_name(),
        )
        self.sm_runtime = assumed_role_session.client(
            "sagemaker-runtime",
            config=Config(  # https://botocore.amazonaws.com/v1/documentation/api/latest/reference/config.html
                region_name=self.region,
                connect_timeout=0.2,  # 200ms
                max_pool_connections=100,
                retries={"total_max_attempts": 3},  # 1 initial + 2 retries
            ),
        )

        self.history: list[dict[str, Any]] = []

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        if not re.match(r"Human: .*\nAssistant:", prompt, flags=re.DOTALL):
            prompt = f"Human: {prompt}\nAssistant:"

        if self.model_type == "chat":
            # TODO: implement chat-style LLM request
            # # caching mechanism requires hashable kwargs
            # kwargs["messages"] = [{"role": "user", "content": prompt}]
            # kwargs = {"stringify_request": json.dumps(kwargs)}
            # response = chat_request(**kwargs)
            raise NotImplementedError()
        else:
            kwargs["prompt"] = prompt
            response = completions_request(self.sm_runtime, **_serialize_values(kwargs))

        history = {
            # "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    @backoff.on_exception(
        backoff.constant,  # Bedrock usually limits 60 TPM
        ERRORS,
        max_tries=3,
        on_backoff=backoff_hdlr,
    )
    def request(self, prompt: str, **kwargs):
        """Handles rate limiting and caching."""
        if "model_type" in kwargs:
            del kwargs["model_type"]

        return self.basic_request(prompt, **kwargs)

    # def _get_choice_text(self, choice: dict[str, Any]) -> str:
    #     if self.model_type == "chat":
    #         return choice["message"]["content"]
    #     return choice["text"]

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        # """Retrieves completions from GPT-3.

        # Args:
        #     prompt (str): prompt to send to GPT-3
        #     only_completed (bool, optional): return only completed responses and ignores completion due to length. Defaults to True.
        #     return_sorted (bool, optional): sort the completion choices using the returned probabilities. Defaults to False.

        # Returns:
        #     list[dict[str, Any]]: list of completion choices
        # """

        # assert only_completed, "for now"
        # assert return_sorted is False, "for now"

        # # if kwargs.get("n", 1) > 1:
        # #     if self.model_type == "chat":
        # #         kwargs = {**kwargs}
        # #     else:
        # #         kwargs = {**kwargs, "logprobs": 5}

        response = self.request(prompt, **kwargs)

        # choices = response["choices"]

        # completed_choices = [c for c in choices if c["finish_reason"] != "length"]

        # if only_completed and len(completed_choices):
        #     choices = completed_choices

        # completions = [self._get_choice_text(c) for c in choices]
        # if return_sorted and kwargs.get("n", 1) > 1:
        #     scored_completions = []

        #     for c in choices:
        #         tokens, logprobs = (
        #             c["logprobs"]["tokens"],
        #             c["logprobs"]["token_logprobs"],
        #         )

        #         if "<|endoftext|>" in tokens:
        #             index = tokens.index("<|endoftext|>") + 1
        #             tokens, logprobs = tokens[:index], logprobs[:index]

        #         avglog = sum(logprobs) / len(logprobs)
        #         scored_completions.append((avglog, self._get_choice_text(c)))

        #     scored_completions = sorted(scored_completions, reverse=True)
        #     completions = [c for _, c in scored_completions]

        # return completions


def _deserialize_values(d):
    return {k: json.loads(v) for k, v in d.items()}


def _serialize_values(d):
    return {k: json.dumps(v) for k, v in d.items()}


def _get_nested_str(inference_result: Union[str, List, Dict]) -> str:
    if isinstance(inference_result, list):
        return _get_nested_str(inference_result[0])
    elif isinstance(inference_result, dict):
        return _get_nested_str(next(iter(inference_result.items()))[1])
    else:
        return inference_result


@CacheMemory.cache
def cached_bedrock_completion_request(sm_runtime, **kwargs):
    kwargs = _deserialize_values(kwargs)
    sm_response = sm_runtime.invoke_endpoint(
        EndpointName=kwargs["endpoint_name"],
        ContentType="application/json",
        Body=json.dumps({"inputs": kwargs["prompt"], "parameters": kwargs["model_config"]}),
    )
    body = sm_response["Body"].read().decode("utf-8")
    return _get_nested_str(json.loads(body))

@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def completions_request(sm_runtime, **kwargs):
    return cached_bedrock_completion_request(**kwargs)
