#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import os
from typing import Any, Dict, List

from jinja2 import Template

from utils.file_utils import load_json

PROMPT_LIB_FILE = os.path.join(os.path.dirname(__file__), "prompt_library.json")


class Prompt:
    def __init__(
        self,
        name: str,
        template_str: str,
        description: str | None = None,
        required_args: List[str] | None = None,
    ):
        self.name = name
        self.template = Template(template_str)
        self.description = description
        self.required_args = required_args if required_args else []

    def prompt_render(self, **kwargs: Dict[str, Any]):
        assert all(
            x in kwargs for x in self.required_args
        ), f"Not all required args ({self.required_args}) are in kwargs ({str(kwargs)})"
        return self.template.render(**kwargs)


def get_prompt(name: str, **kwargs: Dict[str, Any]) -> str:
    prompt_templates = load_json(PROMPT_LIB_FILE)

    if name and name in prompt_templates:
        prompt = Prompt(name=name, **prompt_templates[name])
        return prompt.prompt_render(**kwargs)
    else:
        raise ValueError(f"Prompt '{name}' not found.")
