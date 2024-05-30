if [ ${CKPT_PRE} ]; then
    echo "CKPT: ${CKPT_PRE}"
else
    echo "CKPT not set. Exiting..."
    exit 1
fi

if [ ${TAG_PRE} ]; then
    echo "TAG: ${TAG_PRE}"
else
    echo "TAG not set. Exiting..."
fi

if [ ${BENCHMARKS} ]; then
    echo "Benchmarks: ${BENCHMARKS}"
    BENCHMARKS_arr=(${BENCHMARKS//,/ })
else
    default_benchmarks=("chair_sub" "mme" "pope")
    echo "Benchmarks not set. Using default benchmarks: ${default_benchmarks[*]}"
    BENCHMARKS_arr=${default_benchmarks[*]}
fi

if [ ${SUFFIX} ]; then
    echo "SUFFIX: ${SUFFIX}"
    SUFFIX_arr=(${SUFFIX//,/ })
else
    echo "SUFFIX not set"
fi
source scripts/eval/config.sh

if [ ${BATCH_SIZE} ]; then
    echo "BATCH_SIZE: ${BATCH_SIZE}"
fi

if [ ${SQL_HOST} ]; then
    echo "SQL_HOST: ${SQL_HOST}"
    echo "SQL_PORT: ${SQL_PORT}"
    echo "SQL_USER: ${SQL_USER}"
    echo "SQL_PASSWORD: ${SQL_PASSWORD}"
    echo "SQL_DB: ${SQL_DB}"
fi

declare -A eval_scripts=(
    ["mme"]="scripts/eval/mme.sh"
    ["mme_sgl"]="scripts/eval/mme_sgl.sh"
    ["pope"]="scripts/eval/pope.sh"
    ["pope_sgl"]="scripts/eval/pope_sgl.sh"
    ["mmvet"]="scripts/eval/mmvet.sh"
    ["mmbench"]="scripts/eval/mmbench.sh"
    ["seedbench_gen"]="scripts/eval/seedbench_generate.sh"
    ["mmmu"]="scripts/eval/mmmu.sh"
    ["mathvista"]="scripts/eval/mathvista.sh"
)
mkdir -p ./eval/all
if [ ${SUFFIX} ]; then
    for suf in ${SUFFIX_arr[*]}
    do
        for bench in ${BENCHMARKS_arr[*]}
        do
            export CKPT="${CKPT_PRE}${suf}"
            export TAG="${TAG_PRE}${suf}"
            log_name="./eval/all/${TAG}_${bench}.log"
            echo "---------------------Running $bench on $TAG---------------------"
            source ${eval_scripts[$bench]} > $log_name
        done
    done
else
    for bench in ${BENCHMARKS_arr[*]}
    do
        export CKPT="${CKPT_PRE}"
        export TAG="${TAG_PRE}"
        log_name="./eval/all/${TAG}_${bench}.log"
        echo "---------------------Running $bench on $TAG---------------------"
        source ${eval_scripts[$bench]} > $log_name
    done
fi
