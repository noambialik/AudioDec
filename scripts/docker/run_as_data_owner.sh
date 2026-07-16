#!/usr/bin/env bash
set -euo pipefail

data_root=/workspace/data

if [[ ! -d "${data_root}" ]]; then
  echo "Required data volume is missing: ${data_root}" >&2
  exit 1
fi

owner_uid="$(stat -c '%u' "${data_root}")"
owner_gid="$(stat -c '%g' "${data_root}")"
if [[ "${owner_uid}" == "0" || "${owner_gid}" == "0" ]]; then
  echo "Data volume must not be root-owned: ${data_root}" >&2
  exit 1
fi

if ! getent group "${owner_gid}" >/dev/null; then
  groupadd --gid "${owner_gid}" audiodec
fi
if ! getent passwd "${owner_uid}" >/dev/null; then
  useradd --uid "${owner_uid}" --gid "${owner_gid}" --home-dir /tmp \
    --no-create-home --shell /usr/sbin/nologin audiodec
fi

exec gosu "${owner_uid}:${owner_gid}" "$@"
