# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
from typing import NamedTuple  # , Required, NotRequired
from typing import Any, Dict, List, Optional, TypedDict

import aiohttp
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

try:
    from matrix import Cli
    from matrix.client import query_llm
    from matrix.client.endpoint_cache import EndpointCache

    MATRIX_AVAIL = True
except ModuleNotFoundError:
    Cli = None
    RayCluster = None
    query_llm = None
    EndpointCache = int

    print(
        "Matrix is not found in installation, using legacy redis registry instead, "
        "make sure you have .registry_url.txt in the project root dir."
    )

    MATRIX_AVAIL = False

from utils.file_utils import hash_uuid_to_int
from utils.http_utils import (
    CHAT_ENDPOINT,
    COMP_ENDPOINT,
    DEFAULT_REQUEST_TIMEOUT,
    RayEndpoint,
    get_registry_url_from_file,
    get_worker_list,
    http_post_with_retry,
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)


class ChatTurn(TypedDict):
    role: str
    content: str
    info: Dict[str, Any]


class ChatRole(NamedTuple):
    name: str
    ray_endpoint: RayEndpoint
    sys_prompt: str
    sampling_temp: float
    conv_id: Optional[str]


class ChatRoleFactory:
    """
    The reason we need a factory is because we need to generate the same role based on
    different urls so we can distribute the generation of the conversations.
    """

    def __init__(
        self,
        name: str,
        model_name: str,
        sampling_temp: float,
        app_name: Optional[
            str
        ] = None,  # app_name shall not be Optional once fully migrate to Matrix
        ray_cluster_id: Optional[str] = None,
        ray_rdv_dir: Optional[str] = None,
        sys_prompt: Optional[str] = None,
    ):
        self.name = name
        self.app_name = app_name if app_name else ""
        self.model_name = model_name
        self.sampling_temp = sampling_temp
        self.sys_prompt = sys_prompt

        try:
            registry_url = get_registry_url_from_file()
            self.model_eps = get_worker_list(registry_url, self.model_name)  # type: ignore
        except Exception:
            print(".registry_url.txt is not set or cannot get any endpoint")
            self.model_eps = None

        if self.model_eps is None and MATRIX_AVAIL and Cli is not None:
            metadata = Cli(
                cluster_id=ray_cluster_id, matrix_dir=ray_rdv_dir
            ).get_app_metadata(app_name=self.app_name, endpoint_ttl_sec=30)
            self.endpoint_cache = metadata["endpoints"]["updater"]  # type: ignore

        assert (
            self.model_eps or self.endpoint_cache
        ), "has to get some endpoints either from registry_url or Matrix app"

    def set_role_system_prompt(self, prompt: str):
        self.sys_prompt = prompt

    async def get_chat_role(
        self, sys_prompt: str | None = None, uuid: str | None = None
    ) -> ChatRole:
        if self.endpoint_cache:
            url = ""  # dummy url, will find endpoint in matrix
        else:
            assert self.model_eps is not None

            if uuid:
                model_ep = self.model_eps[hash_uuid_to_int(uuid) % len(self.model_eps)]
            else:
                model_ep = random.choice(self.model_eps)
            url = model_ep.url.replace(
                CHAT_ENDPOINT, "/v1"
            )  # manipulate string for migration, will deprecate

        # TODO: this is not really RayEndpoint yet, need to deprecate model_eps
        ray_ep = RayEndpoint(url, self.app_name, self.model_name)

        if not sys_prompt:
            sys_prompt = self.sys_prompt

        assert sys_prompt is not None, "System prompt must be set!"

        return ChatRole(self.name, ray_ep, sys_prompt, self.sampling_temp, uuid)


def apply_chat_template(
    msg_list: List[ChatTurn],
    sys_prompt: str | None = None,
    last_role_name: str = "user",
) -> List[Dict[str, str]]:
    """
    Generate chat format that strictly follows the pattern of
    "system ; user ; assistant ; user ;..."
    becuase this is the only format that most chat models accept
    """
    assert last_role_name in [
        "user",
        "assistant",
    ], "The last role must be one of `user` or `assistant`"

    if sys_prompt:
        chat_turn_list = [dict(role="system", content=sys_prompt)]
    else:
        chat_turn_list = []

    for i, msg in enumerate(msg_list):
        if (len(msg_list) - i) % 2 == int(last_role_name == "user"):
            chat_turn_list.append(dict(role="user", content=msg["content"]))
        else:
            chat_turn_list.append(dict(role="assistant", content=msg["content"]))

    return chat_turn_list


async def generate_completion(
    chat_turns: List[ChatTurn],
    chat_role: ChatRole,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    append_prompt: str,
    session: aiohttp.ClientSession,
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
    max_tokens: int | None = None,
    n_generations: int = 1,
) -> List[ChatTurn]:
    templated_turns = apply_chat_template(
        chat_turns, sys_prompt=chat_role.sys_prompt, last_role_name="assistant"
    )

    # append the additional prompt
    templated_turns[-1]["content"] += append_prompt
    prompt = tokenizer.apply_chat_template(
        templated_turns, add_generation_prompt=False, tokenize=False
    )
    # eos_token will be appended to the end of the turn by default, so remove it
    prompt = prompt.rstrip(tokenizer.eos_token)  # type: ignore

    request_data = {
        "model": chat_role.ray_endpoint.model_name,
        "prompt": prompt,
        "temperature": chat_role.sampling_temp,
        "n": n_generations,
    }

    if max_tokens is not None:
        request_data["max_tokens"] = max_tokens

    # send the request with aiohttp
    http_response = await http_post_with_retry(
        session=session,
        url=chat_role.ray_endpoint.url.replace("/v1", COMP_ENDPOINT),
        data=request_data,
        timeout=request_timeout,
    )

    if status := http_response["status_ok"]:
        chat_turns = [
            ChatTurn(
                role=chat_role.name,
                content=http_response["info"]["choices"][i]["text"],
                info={"status_ok": status, "usage": http_response["info"]["usage"]},
            )
            for i in range(n_generations)
        ]
    else:
        chat_turns = [
            ChatTurn(
                role=chat_role.name,
                content=http_response["info"]["error"],
                info={"status_ok": status, "error": http_response["info"]["error"]},
            )
            for _ in range(n_generations)
        ]

    return chat_turns


async def generate_next_turn(
    chat_turns: List[ChatTurn],
    chat_role: ChatRole,
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
    max_tokens: int | None = None,
    use_matrix: bool = False,
    endpoint_cache: Any | None = None,
) -> ChatTurn:
    return (
        await generate_next_turn_candidates(
            chat_turns,
            chat_role,
            1,
            request_timeout,
            max_tokens,
            use_matrix,
            endpoint_cache,
        )
    )[0]


async def generate_next_turn_candidates(
    chat_turns: List[ChatTurn],
    chat_role: ChatRole,
    n_generations: int = 1,
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
    max_tokens: int | None = None,
    use_matrix: bool = False,
    endpoint_cache: EndpointCache | None = None,  # type: ignore
) -> List[ChatTurn]:
    """
    Generate the next turn of the chat based on the role and the prev turns.
    NOTE: system prompt is in the `chat_role`.
    """

    # prepare the chat in the way that the chat API expects
    templated_turns = apply_chat_template(chat_turns, sys_prompt=chat_role.sys_prompt)
    if use_matrix:
        assert MATRIX_AVAIL and query_llm is not None

        params = {}
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if chat_role.conv_id is not None:
            params["multiplexed_model_id"] = chat_role.conv_id
        if n_generations > 1:
            params["n"] = n_generations

        result = await query_llm.make_request(
            url=chat_role.ray_endpoint.url,
            model=chat_role.ray_endpoint.model_name,
            app_name=chat_role.ray_endpoint.app_name,
            data={"messages": templated_turns},
            temperature=chat_role.sampling_temp,
            timeout_secs=request_timeout,
            endpoint_cache=endpoint_cache,
            **(params),
        )
        if "text" in result["response"]:
            chat_turn_candidates = [
                ChatTurn(
                    role=chat_role.name,
                    content=result["response"]["text"][i],
                    info={"status_ok": True, "usage": result["response"]["usage"]},
                )
                for i in range(n_generations)
            ]
        else:
            chat_turn_candidates = [
                ChatTurn(
                    role=chat_role.name,
                    content=result["response"]["error"],
                    info={"status_ok": False, "error": result["response"]["error"]},
                )
                for i in range(n_generations)
            ]
    else:
        request_data = {
            "model": chat_role.ray_endpoint.model_name,
            "messages": templated_turns,
            "temperature": chat_role.sampling_temp,
        }

        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens
        if n_generations > 1:
            request_data["n"] = n_generations

        # send the request with aiohttp
        headers = (
            {"serve_multiplexed_model_id": chat_role.conv_id}
            if chat_role.conv_id
            else None
        )
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=request_timeout)
        ) as session:
            http_response = await http_post_with_retry(
                session=session,
                url=chat_role.ray_endpoint.url.replace("/v1", CHAT_ENDPOINT),
                data=request_data,
                timeout=request_timeout,
                headers=headers,
            )

        if status := http_response["status_ok"]:
            chat_turn_candidates = [
                ChatTurn(
                    role=chat_role.name,
                    content=http_response["info"]["choices"][i]["message"]["content"],
                    info={"status_ok": status, "usage": http_response["info"]["usage"]},
                )
                for i in range(n_generations)
            ]
        else:
            chat_turn_candidates = [
                ChatTurn(
                    role=chat_role.name,
                    content=http_response["info"]["error"],
                    info={"status_ok": status, "error": http_response["info"]["error"]},
                )
                for _ in range(n_generations)
            ]

    return chat_turn_candidates
