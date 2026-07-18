#!/usr/bin/env bash
set -euo pipefail

started_at_seconds="$(date +%s)"

explicit_config=false
for argument in "$@"; do
  case "${argument}" in
    -c|--config|--config=*) explicit_config=true ;;
  esac
done

training_args=("$@")
if [[ "${explicit_config}" == false ]]; then
  training_args=(
    --config /workspace/AudioDec/config/autoencoder/symAD_libritts_24000_hop300.yaml
    --tag autoencoder/symAD_libritts_24000_hop300
    "$@"
  )
fi

run_tag="$(python -c 'import sys; from bin.utils import load_training_config, resolve_experiment_tag; args = sys.argv[1:]; value = lambda names: next((argument.split("=", 1)[1] if "=" in argument else args[index + 1] for index, argument in enumerate(args) if argument.split("=", 1)[0] in names), None); print(resolve_experiment_tag(load_training_config(value(("-c", "--config"))), value(("--tag",))))' "${training_args[@]}")"

set +e
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"; assert torch.cuda.device_count() == 4, f"expected 4 mapped GPUs, found {torch.cuda.device_count()}"; print("Mapped GPUs:", ", ".join(f"cuda:{index}={torch.cuda.get_device_name(index)}" for index in range(torch.cuda.device_count())))' && \
python /workspace/AudioDec/codecTrain.py \
    --exp_root /workspace/data/audiodec/exp \
    "${training_args[@]}"
training_status=$?
set -e

if [[ "${training_status}" -eq 0 ]]; then
  result=finished
  status=success
else
  result=failed
  status=failed
fi
duration_seconds="$(( $(date +%s) - started_at_seconds ))"
printf -v body 'status=%s\ntag=%s\nhost=%s\nduration_seconds=%s\n' \
  "${status}" "${run_tag}" "$(hostname)" "${duration_seconds}"

if ! python -c 'import sys; from urllib import request; req = request.Request("https://ntfy.sh/noamb_audiodec", data=sys.argv[2].encode("utf-8"), method="POST", headers={"Title": sys.argv[1], "Content-Type": "text/plain; charset=utf-8"}); request.urlopen(req, timeout=10).close()' \
  "AudioDec LibriTTS training ${result}" "${body}"; then
  echo "ntfy notification failed; preserving training exit status ${training_status}." >&2
fi

exit "${training_status}"
