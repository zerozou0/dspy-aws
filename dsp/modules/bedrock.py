import functools
import json
import re
from typing import Any, Dict, Literal, Optional, Union

import backoff
import boto3
import ipdb

from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on
from dsp.modules.lm import LM


# TODO: Make BEDROCK_RT_CLIENT configurable
RETRYABLE_ERRORS = tuple([])
REGION = "us-east-1"
BEDROCK_RT_CLIENT = boto3.client(
    "bedrock-runtime", REGION, endpoint_url=f"https://bedrock.{REGION}.amazonaws.com"
)


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details)
    )


class Bedrock(LM):
    """Wrapper around AWS Bedrock API.

    Args:
        model (str, optional): OpenAI or Azure supported LLM model to use. Defaults to "text-davinci-002".
        api_key (Optional[str], optional): API provider Authentication token. use Defaults to None.
        api_provider (Literal["openai", "azure"], optional): The API provider to use. Defaults to "openai".
        model_type (Literal["chat", "text"], optional): The type of model that was specified. Mainly to decide the optimal prompting strategy. Defaults to "text".
        **kwargs: Additional arguments to pass to the API provider.
    """

    DEFAULT_GEN_CONFIG = {
        "max_tokens_to_sample": 500,
        "temperature": 0,
        "top_k": 1,
        "stop_sequences": ["\n\nHuman:"],
    }
    BR_GEN_PARAM_CONVERSION = {
        # other name: bedrock name
        "max_tokens": "max_tokens_to_sample",
        "temperature": "temperature",
    }
    RETAINED_LM_KWARGS = (
        "model",
        "n",  # Indicating batch size
    )

    def __init__(
        self,
        model_id: str = "anthropic.claude-v2",
        # invocation_role_arn: str,
        model_config: Optional[Dict] = None,
        model_type: Literal["chat", "text"] = "text",
        **kwargs,
    ):
        super().__init__(model_id)
        self.provider = "bedrock"
        
        self.model_id = model_id
        self.model_type = model_type
        self.model_config = (
            self.DEFAULT_GEN_CONFIG if not model_config else
            {**self.DEFAULT_GEN_CONFIG, **model_config}
        )

        # Clean up kwargs set by parent
        # keys_to_remove = []
        # for key in self.kwargs:
        #     # Mirror the actual Bedrock gen config values to self.kwargs for consumption by other DSPy fns
        #     if (
        #         self.BR_GEN_PARAM_CONVERSION.get(key) not in self.model_config or
        #         key not in self.RETAINED_LM_KWARGS
        #     ):
        #         keys_to_remove.append(key)
        #     else:
        #         self.kwargs[key] = self.model_config[self.BR_GEN_PARAM_CONVERSION[key]]
        self.kwargs = {
            "model_id": self.model_id,
            # "model_config": self.model_config,
            # **{k: v for k, v in self.kwargs.items() if k not in keys_to_remove}
            **self._get_synced_kwargs(self.model_config)
        }

        self.history: list[dict[str, Any]] = []

    def basic_request(self, prompt: str, **gen_param_override) -> Dict[str, Any]:
        """
        gen_param_override are limited to model generation config params only.
        """
        raw_kwargs = gen_param_override
        final_gen_params = {
            **self.model_config, 
            **self._remove_non_br_kwargs(gen_param_override)
        }
        # if not re.match(r"Human: .*\nAssistant:", prompt, flags=re.DOTALL):
        #     prompt = f"Human: {prompt}\nAssistant:"
        if not re.match(r"Human:", prompt, flags=re.DOTALL):
            prompt = f"Human: {prompt}\n\nAssistant:\nReasoning: Let's think step by step in order to"

        if self.model_type == "chat":
            # TODO: implement chat-style LLM request
            # # caching mechanism requires hashable kwargs
            # kwargs["messages"] = [{"role": "user", "content": prompt}]
            # kwargs = {"stringify_request": json.dumps(kwargs)}
            # response = chat_request(**kwargs)
            kwargs_for_caching = self._get_synced_kwargs(final_gen_params)
            raise NotImplementedError()
        else:
            final_gen_params["prompt"] = prompt
            kwargs_for_caching = self._get_synced_kwargs(final_gen_params)
            completion_text = completions_request(
                **_serialize_values(kwargs_for_caching)
            )["completion"]

        # Wrap in choices / text for history inspection
        response = {"choices": [{"text": completion_text}]}
        # History recording should happen right after the cache call wrappers
        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs_for_caching,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return completion_text

    # @backoff.on_exception(
    #     backoff.constant,  # Bedrock usually limits 60 TPM
    #     ERRORS,
    #     max_tries=3,
    #     on_backoff=backoff_hdlr,
    # )
    # def request(self, prompt: str, **kwargs):
    #     """Handles rate limiting and caching."""
    #     if "model_type" in kwargs:
    #         del kwargs["model_type"]
    #     return self.basic_request(prompt, **kwargs)

    @backoff.on_exception(
        backoff.constant,  # Bedrock usually limits 60 TPM
        RETRYABLE_ERRORS,
        max_tries=3,
        on_backoff=backoff_hdlr,
    )
    def __call__(
        self,
        prompt: str,
        **kwargs,
    ) -> list[str]:
        """Retrieves completions from Bedrock.

        Args:
            prompt (str): prompt to send to Bedrock

        Returns:
            List: list of completion choices
        """
        # ipdb.set_trace()
        return [self.basic_request(prompt, **kwargs)]
    
    def _get_synced_kwargs(self, new_model_config):
        # Synced for calling the cached completion fn
        completion_kwargs = {"model_config": new_model_config,}
        for key, value in self.kwargs.items():
            if key == "model_config":
                continue
            elif self.BR_GEN_PARAM_CONVERSION.get(key) not in new_model_config:
                completion_kwargs[key] = value
            else:
                completion_kwargs[key] = new_model_config[self.BR_GEN_PARAM_CONVERSION[key]]
        return completion_kwargs
    
    def _remove_non_br_kwargs(self, gen_param_override):
        keys_to_remove = []
        for k, v in gen_param_override.items():
            if k in self.BR_GEN_PARAM_CONVERSION:
                if self.BR_GEN_PARAM_CONVERSION[k] in gen_param_override:
                    gen_param_override[self.BR_GEN_PARAM_CONVERSION[k]] = v
                keys_to_remove.append(k)
            elif k in self.RETAINED_LM_KWARGS:
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del gen_param_override[k]
        return gen_param_override


def _deserialize_values(d):
    return {k: json.loads(v) for k, v in d.items()}


def _serialize_values(d) -> Dict[str, str]:
    return {k: json.dumps(v) for k, v in d.items()}


# def _filter_generation_params(d):
#     return {
#         "prompt": d["prompt"],
#         **d["model_config"],
#     }


# def _get_nested_str(inference_result: Union[str, List, Dict]) -> str:
#     if isinstance(inference_result, list):
#         return _get_nested_str(inference_result[0])
#     elif isinstance(inference_result, dict):
#         return _get_nested_str(next(iter(inference_result.items()))[1])
#     else:
#         return inference_result


@CacheMemory.cache
def cached_bedrock_completion_request(**kwargs) -> Dict[str, Any]:
    kwargs = _deserialize_values(kwargs)
    # ipdb.set_trace()
    br_response = BEDROCK_RT_CLIENT.invoke_model(
        modelId=kwargs["model_id"],
        contentType="application/json",
        accept="application/json",
        body=json.dumps(kwargs["model_config"]),
    )
    return json.loads(br_response["body"].read().decode("utf-8"))


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def completions_request(**kwargs) -> Dict[str, Any]:  # kwargs has to be str values
    # ipdb.set_trace()
    return cached_bedrock_completion_request(**kwargs)
