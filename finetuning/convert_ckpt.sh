# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash

SIZE=$1
CKPT_DIR=$2

case "$SIZE" in
    "8b")
        n_shards=1
        ;;
    "70b")
        n_shards=8
        ;;
    *)
        echo "$var is not in the list"
        ;;
esac

check_and_print_env_var() {
  local var_name="$1"

  # Check if the variable is set and exported
  if [ -z "${!var_name+x}" ]; then
    echo "Error: Environment variable '$var_name' is not set or exported." >&2
    return 1
  else
    echo "$var_name is set to: ${!var_name}"
  fi
}

# tokenizer dir need to be manually set with env var
check_and_print_env_var tokenizer_dir

# copy the tokenizers and link the weights
mkdir -p ${CKPT_DIR}_hf_converted
touch ${CKPT_DIR}_hf_converted/fs2_native
ln -s ${CKPT_DIR}/config.json ${CKPT_DIR}_hf_converted/config.json

if [ "$n_shards" == 1 ]; then
    ln -s ${CKPT_DIR}/model.pt ${CKPT_DIR}_hf_converted/model.pt
else
    for i in $(seq 0 $((n_shards - 1))); do
        ln -s ${CKPT_DIR}/model.$i.pt ${CKPT_DIR}_hf_converted/model.$i.pt
    done
fi

echo "Lastly copy the tokenizers and configs back to the hf folder"
cp $tokenizer_dir/*.json ${CKPT_DIR}_hf_converted
