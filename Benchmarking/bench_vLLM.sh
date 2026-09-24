# bench.sh
URL="https://your-workspace--vllm-inference-for-web-search-agent.modal.run"
for CONCURRENCY in 1 8 32 64; do
  vllm bench serve \
    --backend openai-chat \
    --base-url "$URL" \
    --model IFM/K2-Horizon-32B \
    --dataset-name random \
    --random-input-len 2000 --random-output-len 200 \
    --num-prompts 128 \
    # --request-rate 2 \
    --max-concurrency $CONCURRENCY \
    --ignore-eos --seed 0 \
    --save-result --result-filename "run_c${CONCURRENCY}.json"\
    --seed 411
done
# need to log config and (throughput, TTFT, TPOT, ITL).