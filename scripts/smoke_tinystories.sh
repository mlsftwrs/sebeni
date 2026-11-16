#!/usr/bin/env bash
# Smoke-run every Sebeni arm on roneneldan/TinyStories-1M.
#
# Usage (from the repository root):
#   scripts/smoke_tinystories.sh                 # all arms, GPU if available, else CPU
#   scripts/smoke_tinystories.sh sft grpo        # chosen arms only
#   SEBENI_FORCE_CPU=1 scripts/smoke_tinystories.sh
#   PYTHON=/path/to/venv/bin/python scripts/smoke_tinystories.sh
#   KVERITAS_MODE=local|server|off scripts/smoke_tinystories.sh   # default: local
#
# SFT and GRPO run distill -> train -> eval; DPO and APO run `sebeni exp`.
# Each arm gets its own working directory under ${SMOKE_ROOT:-${TMPDIR:-/tmp}/sebeni-smoke-tinystories}.
# K-Veritas (#20): every train / eval / exp command runs under `kveritas run`.
#   local  (default) `kveritas init --local`, signed with a throwaway RSA key in SMOKE_ROOT/keys
#   server the config as is (attestation server)
#   off    experiment.kveritas / kveritas_seal set to false
# Each sealed report (exp: exp/report.pdf; train/eval: exp/report_{train,eval}.pdf) is checked with `kveritas verify`.
# Checks pipeline mechanics only: TinyStories is an English story model.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Run the CLI from this checkout (PYTHONPATH), never from an installed sebeni copy.
PYTHON="${PYTHON:-$(command -v python || command -v python3)}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
sebeni() { "${PYTHON}" -c 'from beni.cli.main import app; app()' "$@"; }
"${PYTHON}" -c 'import beni, sys; sys.exit(0 if beni.__file__.startswith(sys.argv[1]) else 1)' "${REPO_ROOT}" || {
  echo "beni does not import from ${REPO_ROOT}; check PYTHON=${PYTHON}" >&2; exit 2; }

SMOKE_ROOT="${SMOKE_ROOT:-${TMPDIR:-/tmp}/sebeni-smoke-tinystories}"
CONFIG="${REPO_ROOT}/configs/test_tinystories.yaml"
mkdir -p "${SMOKE_ROOT}"
SMOKE_CONFIG="${SMOKE_ROOT}/test_tinystories.smoke.yaml"
cp "${CONFIG}" "${SMOKE_CONFIG}"
if [[ "${SEBENI_FORCE_CPU:-0}" == "1" ]]; then
  sed -i 's/use_cpu: false/use_cpu: true/' "${SMOKE_CONFIG}"
fi
KVERITAS_MODE="${KVERITAS_MODE:-local}"
case "${KVERITAS_MODE}" in
  local)
    KEY="${SMOKE_ROOT}/keys/private.pem"
    if [[ ! -f "${KEY}" ]]; then
      mkdir -p "$(dirname "${KEY}")" && openssl genrsa -out "${KEY}" 3072 2>/dev/null && chmod 600 "${KEY}"
    fi
    sed -i "s|^  kveritas_seal: true|  kveritas_seal: true\n  kveritas_local: true\n  kveritas_local_key: ${KEY}|" "${SMOKE_CONFIG}"
    ;;
  server) ;;
  off) sed -i 's/^  kveritas: true/  kveritas: false/; s/^  kveritas_seal: true/  kveritas_seal: false/' "${SMOKE_CONFIG}" ;;
  *) echo "KVERITAS_MODE must be local, server or off" >&2; exit 2 ;;
esac
CONFIG="${SMOKE_CONFIG}"

DEVICE="$("${PYTHON}" -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")' 2>/dev/null || echo unknown)"
[[ "${SEBENI_FORCE_CPU:-0}" == "1" ]] && DEVICE="cpu (forced)"
echo "[smoke] python=${PYTHON} config=${CONFIG} device=${DEVICE} kveritas=${KVERITAS_MODE} root=${SMOKE_ROOT}"

ARMS=("$@")
[[ ${#ARMS[@]} -eq 0 ]] && ARMS=(sft grpo dpo apo)

declare -A RESULT
for arm in "${ARMS[@]}"; do
  wd="${SMOKE_ROOT}/${arm}"
  rm -rf "${wd}"
  start=$(date +%s)
  log="${SMOKE_ROOT}/${arm}.log"
  echo "[smoke] ${arm}: log ${log}"
  if {
    case "${arm}" in
      sft|grpo)
        sebeni distill -c "${CONFIG}" -w "${wd}" &&
        sebeni train -c "${CONFIG}" -w "${wd}" --algorithm "${arm}" &&
        sebeni eval -c "${CONFIG}" -w "${wd}"
        ;;
      dpo|apo)
        sebeni exp -c "${CONFIG}" -w "${wd}" --algorithm "${arm}"
        ;;
      *)
        echo "unknown arm: ${arm}" >&2; false
        ;;
    esac
  } >"${log}" 2>&1; then
    status=ok
  else
    status=FAILED
  fi
  secs=$(( $(date +%s) - start ))
  metrics="$(python3 - "${wd}/exp/eval.json" <<'PY' 2>/dev/null || echo "no eval.json"
import json, sys
r = json.load(open(sys.argv[1]))
print(" ".join(f"{k}={r.get(k)}" for k in ("phi", "mer", "mcs", "uwec")))
PY
)"
  kvstatus=""
  if [[ "${KVERITAS_MODE}" != "off" ]]; then
    # exp seals one report; the train -> eval flow seals one per command.
    case "${arm}" in
      sft|grpo) reports=(report_train.pdf report_eval.pdf) ;;
      *) reports=(report.pdf) ;;
    esac
    for report in "${reports[@]}"; do
      path="${wd}/exp/${report}"
      if [[ -f "${path}" ]] && command -v kveritas >/dev/null; then
        runs="$(kveritas verify "${path}" 2>&1 | awk '/^Runs:/ {print $2}')"
        if kveritas verify "${path}" >/dev/null 2>&1; then
          kvstatus+=" ${report%.pdf}=verified(runs=${runs})"
        else
          kvstatus+=" ${report%.pdf}=VERIFY_FAILED"; status=FAILED
        fi
      else
        kvstatus+=" ${report%.pdf}=MISSING"; status=FAILED
      fi
    done
  fi
  RESULT[${arm}]="${status} ${secs}s ${metrics}${kvstatus}"
  echo "[smoke] ${arm}: ${RESULT[${arm}]}"
done

echo
echo "[smoke] summary (device=${DEVICE})"
failed=0
for arm in "${ARMS[@]}"; do
  echo "  ${arm}: ${RESULT[${arm}]}"
  [[ "${RESULT[${arm}]}" == FAILED* ]] && failed=1
done
exit "${failed}"
