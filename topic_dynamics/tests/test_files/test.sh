which bedtools &>/dev/null || { echo "ERROR: bedtools not found!"; exit 1; }
>&2 echo "rip.sh: $@"

if [[ $# -lt 2 ]]; then
    echo "Need 2 parameters! <reads> <peaks>"
    exit 1
fi

READS=$1
PEAKS_FILE=$2

RIP_FILE=${PEAKS_FILE}_rip.csv

echo "READS_FILE: ${READS}"
echo "PEAKS_FILE: ${PEAKS_FILE}"
echo "RIP_FILE: ${RIP_FILE}"

# If we already have rip file, do not recalculate
if [[ -f ${RIP_FILE} ]]; then
    echo "${RIP_FILE}"
    cat ${RIP_FILE}
    exit 0
fi

[[ ! -z ${WASHU_ROOT} ]] || { echo "ERROR: WASHU_ROOT not configured"; exit 1; }
source ${WASHU_ROOT}/parallel/util.sh
export TMPDIR=$(type job_tmp_dir &>/dev/null && echo "$(job_tmp_dir)" || echo "/tmp")
mkdir -p "${TMPDIR}"
