"""Run inside the serving container: one greedy, thinking-OFF request.

Uses the target's actual chat template and /generate's strict accepted-draft
counts (bonus tokens do not count as accepted drafts). Does not start/stop a
server or change its configuration. Inspect the printed output for correctness.
"""

import argparse
import json
import os
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--tokenizer", default="/models/target")
    parser.add_argument("--algorithm", choices=("DFLASH", "EAGLE"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def request(path, payload=None):
        headers = {"Content-Type": "application/json"}
        if os.environ.get("SGLANG_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["SGLANG_API_KEY"]
        req = urllib.request.Request(
            args.url.rstrip("/") + path,
            data=None if payload is None else json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.load(response)

    info = request("/server_info")
    assert info["speculative_algorithm"] == args.algorithm, info.get("speculative_algorithm")
    assert info["enable_linear_replayssm_spec"], "ReplaySSM must remain enabled"
    assert info["quantization"] == "auto-round", "Target quantization changed"
    if args.algorithm == "DFLASH":
        assert info["speculative_num_draft_tokens"] == 8, "Use block size 8"
        assert info["speculative_draft_model_quantization"] is None, "Draft must be unquant"
        assert info["speculative_draft_model_path"], "External draft missing"
    else:
        for key, expected in (("speculative_num_steps", 3),
                              ("speculative_eagle_topk", 1),
                              ("speculative_num_draft_tokens", 4)):
            assert info[key] == expected, (key, info[key])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content":
          "Write a Python function that returns the first n Fibonacci numbers, "
          "then briefly explain how it works."}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    if hasattr(input_ids, "input_ids"):
        input_ids = input_ids.input_ids
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0, "top_p": 1, "top_k": 1,
            "min_p": 0, "max_new_tokens": 192,
        },
        "stream": False,
    }
    result = request("/generate", payload)
    record = {"algorithm": args.algorithm, "request": payload, "response": result}
    # Save the evidence before asserting, including on a zero-acceptance run.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(args.output)
    meta = result["meta_info"]
    print(result.get("text", ""))
    print(json.dumps({k: v for k, v in meta.items()
                      if k.startswith("spec_") or k in ("completion_tokens", "finish_reason")},
                     indent=2))
    assert result.get("text", "").strip(), "Empty output"
    assert meta.get("finish_reason", {}).get("type") in ("stop", "length"), meta
    assert meta.get("spec_verify_ct", 0) > 0, "No target verification recorded"
    assert meta.get("spec_num_correct_drafts", 0) > 0, "Zero accepted draft tokens"
    assert meta.get("spec_accept_rate", 0) > 0, "Zero acceptance"
    print(f"PASS: {args.algorithm} verify and accepted-draft counts > 0; evidence: {args.output}")


if __name__ == "__main__":
    main()
