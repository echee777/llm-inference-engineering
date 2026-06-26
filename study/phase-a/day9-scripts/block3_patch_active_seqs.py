#!/usr/bin/env python3
"""Patch pip-installed vLLM scheduler to log active sequences per iteration.

Adds a [SCHED_STEP] log line after each schedule() call showing:
- Timestamp
- Iteration-level active request IDs (short suffixes)
- Count of new, running, finished, preempted requests

Usage:
    python3 /tmp/patch_active_seqs.py /path/to/vllm
"""
import sys
import os
import shutil


def patch_scheduler(vllm_dir):
    filepath = os.path.join(vllm_dir, 'v1', 'core', 'sched', 'scheduler.py')
    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} not found")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    if '[SCHED_STEP]' in content:
        print("  Already patched (SCHED_STEP present)")
        return True

    # Anchor: the line after prev_step_scheduled_req_ids is updated
    anchor = "        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())"

    if anchor not in content:
        print(f"  ERROR: anchor not found")
        return False

    log_block = '''
        # --- SCHED_STEP instrumentation ---
        _new_ids = sorted(r.request_id[-8:] for r in scheduled_new_reqs)
        _resumed_ids = sorted(r.request_id[-8:] for r in scheduled_resumed_reqs)
        _running_ids = sorted(r.request_id[-8:] for r in scheduled_running_reqs)
        _preempted_ids = sorted(r.request_id[-8:] for r in preempted_reqs)
        _finished_ids = sorted(rid[-8:] for rid in self.finished_req_ids)
        _all_active = sorted(set(_new_ids + _resumed_ids + _running_ids))
        logger.info(
            "[SCHED_STEP] ts=%d active=[%s] new=%d running=%d resumed=%d "
            "finished=[%s] preempted=[%s] total_tokens=%d",
            int(time.time() * 1000),
            ",".join(_all_active),
            len(_new_ids),
            len(_running_ids),
            len(_resumed_ids),
            ",".join(_finished_ids),
            ",".join(_preempted_ids),
            total_num_scheduled_tokens,
        )
'''

    content = content.replace(anchor, anchor + log_block)

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"  Patched: {filepath}")
    return True


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 patch_active_seqs.py /path/to/vllm")
        sys.exit(1)

    vllm_dir = sys.argv[1]
    if not os.path.isdir(vllm_dir):
        print(f"ERROR: {vllm_dir} is not a directory")
        sys.exit(1)

    print("=== Active Sequence Logging Patcher ===\n")
    ok = patch_scheduler(vllm_dir)
    if ok:
        print("\nDone! Verify with:")
        print(f"  grep -n 'SCHED_STEP' {vllm_dir}/v1/core/sched/scheduler.py")
    else:
        print("\nPatch failed. Check errors above.")


if __name__ == '__main__':
    main()
