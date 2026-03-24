#!/usr/bin/env bash
#
# Run the full SHMIP test suite (A -> B -> C -> D -> E -> F).
#
# Usage:
#   ./run_all.sh              # run everything
#   ./run_all.sh A B          # run only suites A and B
#   ./run_all.sh D+           # run D, E, F
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

SUITES_ALL=(A B C D E F)

if [[ $# -eq 0 ]]; then
    SUITES=("${SUITES_ALL[@]}")
else
    SUITES=()
    for arg in "$@"; do
        arg_upper="${arg^^}"
        if [[ "${arg_upper}" == *"+" ]]; then
            letter="${arg_upper%+}"
            found=0
            for s in "${SUITES_ALL[@]}"; do
                if [[ "${s}" == "${letter}" ]]; then found=1; fi
                if [[ ${found} -eq 1 ]]; then SUITES+=("${s}"); fi
            done
        else
            SUITES+=("${arg_upper}")
        fi
    done
fi

echo "SHMIP Full Suite Runner"
echo "  Suites: ${SUITES[*]}"
echo "  Logs:   ${LOG_DIR}/"
echo ""

FAILED=()

run_suite() {
    local suite="$1"
    local dir="$2"
    local script="$3"
    shift 3
    local extra_args=("$@")

    local log="${LOG_DIR}/suite_${suite}.log"

    echo "Suite ${suite}"
    echo "  Dir:    ${dir}"
    echo "  Script: ${script}"
    echo "  Log:    ${log}"

    if [[ ! -f "${dir}/${script}" ]]; then
        echo "  ERROR: ${dir}/${script} not found, skipping."
        FAILED+=("${suite}")
        return
    fi

    local start_time=$SECONDS
    if (cd "${dir}" && python3 "${script}" "${extra_args[@]}" 2>&1) | tee "${log}"; then
        local elapsed=$(( SECONDS - start_time ))
        echo ""
        echo "  Suite ${suite} completed in $(( elapsed / 3600 ))h $(( (elapsed % 3600) / 60 ))m $(( elapsed % 60 ))s"
        echo ""
    else
        local elapsed=$(( SECONDS - start_time ))
        echo ""
        echo "  Suite ${suite} FAILED after $(( elapsed / 3600 ))h $(( (elapsed % 3600) / 60 ))m $(( elapsed % 60 ))s"
        echo "  See ${log} for details."
        echo ""
        FAILED+=("${suite}")
    fi
}

for suite in "${SUITES[@]}"; do
    case "${suite}" in
        A)
            run_suite A "${SCRIPT_DIR}/shmip_A" run_shmip_A.py
            ;;
        B)
            run_suite B "${SCRIPT_DIR}/shmip_B" run_shmip_B.py
            ;;
        C)
            if [[ ! -f "${SCRIPT_DIR}/shmip_B/outputs/checkpoints/B5.h5" ]]; then
                echo "  WARNING: B5 checkpoint not found. Suite C may cold-start."
            fi
            run_suite C "${SCRIPT_DIR}/shmip_C" run_shmip_C.py
            ;;
        D)
            if [[ ! -f "${SCRIPT_DIR}/shmip_A/outputs/checkpoints/A1.h5" ]]; then
                echo "  WARNING: A1 checkpoint not found. Suite D may cold-start."
            fi
            run_suite D "${SCRIPT_DIR}/shmip_D" run_shmip_D.py
            ;;
        E)
            run_suite E "${SCRIPT_DIR}/shmip_E" run_shmip_E.py
            ;;
        F)
            if [[ ! -f "${SCRIPT_DIR}/shmip_E/outputs/checkpoints/E1.h5" ]]; then
                echo "  NOTE: E1 checkpoint not found. F0 will cold-start."
            fi
            run_suite F "${SCRIPT_DIR}/shmip_F" run_shmip_F.py
            ;;
        *)
            echo "  Unknown suite: ${suite} (expected A-F), skipping."
            ;;
    esac
done

echo ""
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All suites completed successfully."
else
    echo "FAILED suites: ${FAILED[*]}"
    echo "Check logs in ${LOG_DIR}/ for details."
fi
