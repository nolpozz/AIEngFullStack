# bench.sh
# URL="https://nolpozz--vllm-inference-for-web-search-agent-server-dev.us-east.modal.direct"
# for C in 1 16 64 96 128; do
#   python bench_openai.py \
#     --base-url "$URL" \
#     --model IFM/K2-Horizon-32B \
#     --input-len 2000 --output-len 200 \
#     --num-prompts $((C * 4)) \
#     --max-concurrency "$C" \
#     --seed 411 \
#     --result-filename "run_c${C}.json"
# done

# for CONCURRENCY in 1 8 32 64; do
#   vllm bench serve \
#     --backend openai-chat \
#     --base-url "$URL" \
#     --model IFM/K2-Horizon-32B \
#     --dataset-name random \
#     --random-input-len 2000 --random-output-len 200 \
#     --num-prompts 128 \
#     --max-concurrency $CONCURRENCY \
#     --ignore-eos \
#     --save-result --result-filename "run_c${CONCURRENCY}.json" \
#     --seed 411
# done
# --request-rate 2 \ # Limit of how many of num-prompts come in at a time; can back up
# need to log config and (throughput, TTFT, TPOT, ITL).