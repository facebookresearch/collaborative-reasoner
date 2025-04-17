# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List

from tqdm import tqdm

QUICK_PATH_DICT = {}


def get_quick_path(file_path: str) -> str:
    """Quick path is used to get a quick expansion of the path."""
    if file_path.startswith("#"):
        path = Path(file_path)
        if (qp_head := path.parts[0]) not in QUICK_PATH_DICT:
            raise ValueError(f"Unknown quick path {qp_head}")
        else:
            file_path = str(Path(QUICK_PATH_DICT[qp_head]).joinpath(*path.parts[1:]))

    return file_path


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    file_path = get_quick_path(file_path)

    with open(file_path, "r") as f:
        data = [json.loads(s) for s in f.readlines()]

    return data


def load_json(file_path: str) -> Dict[str, Any]:
    file_path = get_quick_path(file_path)

    with open(file_path, "r") as f:
        data = json.load(f)

    return data


def load_jsonl_line_by_line(
    file_path: str, max_lines: int | None = None
) -> List[Dict[str, Any]]:
    file_path = get_quick_path(file_path)

    with open(file_path, "r") as f:
        data = []
        while line := f.readline():
            if max_lines is not None and len(data) >= max_lines:
                break
            data.append(json.loads(line))

    return data


def load_jsonl_with_progress(file_path: str) -> Iterator[Dict[str, Any]]:
    with open(file_path, "r") as f:
        for line in tqdm(f, desc="Reading JSONL file"):
            yield json.loads(line)


def save_json(file_path: str, json_dict: Dict[str, Any]) -> None:
    file_path = get_quick_path(file_path)

    with open(file_path, "w+") as f:
        json.dump(json_dict, f)


def save_jsonl(file_path: str, data: List[Dict[str, Any]]) -> None:
    file_path = get_quick_path(file_path)

    with open(file_path, "w+") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")


def get_jsonl_files_in_dir(directory):
    # List to store the paths of .jsonl files
    jsonl_files = []
    # Iterate over all files in the specified directory
    for filename in os.listdir(directory):
        # Check if the file ends with .jsonl
        if filename.endswith(".jsonl"):
            # Construct the full file path and add it to the list
            jsonl_files.append(os.path.join(directory, filename))
    return jsonl_files


def hash_uuid_to_int(uuid_value: str) -> int:
    uuid_bytes = uuid.UUID(uuid_value).bytes
    hash_object = hashlib.sha256()
    hash_object.update(uuid_bytes)
    hex_hash = hash_object.hexdigest()
    int_hash = int(hex_hash, 16)

    return int_hash
