#!/usr/bin/env python3
"""
Apply block instrumentation logging to pip-installed vLLM 0.17.1.

What it does: patches vLLM V1's kv_cache_manager.py and scheduler.py in place so
the engine emits one log line per KV-block event (BLOCK_ALLOC, BLOCK_ALLOC_FAIL,
BLOCK_FREE, BLOCK_PREEMPT), each tagged with timestamp, request id, and
free/total block counts. This is the measurement tool the rest of the residency
depends on: it makes every allocation, free, and preemption observable so
capacity and preemption behavior can be measured directly rather than inferred
from aggregate metrics. See ../day8-work.md for the writeup and sample output.

Usage:
    # point it at your installed vLLM package directory (from `pip show vllm`):
    python3 apply_block_instrumentation.py /path/to/site-packages/vllm
"""
import sys
import os
import re
import shutil

def patch_file(filepath, patches):
    """Apply a list of (anchor, insertion, position) patches to a file.

    Each patch is a tuple:
      - anchor: string to find in the file
      - insertion: string to insert
      - position: 'after' or 'before' the anchor line
    """
    with open(filepath, 'r') as f:
        content = f.read()

    for anchor, insertion, position in patches:
        if anchor not in content:
            print(f"  WARNING: Anchor not found in {filepath}:")
            print(f"    {anchor[:80]}...")
            continue

        if insertion.strip() in content:
            print(f"  SKIP: Already patched ({insertion.strip()[:50]}...)")
            continue

        if position == 'after':
            content = content.replace(anchor, anchor + insertion)
        elif position == 'before':
            content = content.replace(anchor, insertion + anchor)

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"  Patched: {filepath}")


def add_import_if_missing(filepath, import_line):
    """Add an import line after existing imports if not already present."""
    with open(filepath, 'r') as f:
        content = f.read()

    if import_line in content:
        print(f"  SKIP: import already present")
        return

    # Add after the last 'import' or 'from' line in the header
    lines = content.split('\n')
    last_import_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            last_import_idx = i

    lines.insert(last_import_idx + 1, import_line)
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Added import: {import_line}")


def patch_kv_cache_manager(vllm_dir):
    """Patch kv_cache_manager.py with BLOCK_ALLOC, BLOCK_ALLOC_FAIL, BLOCK_FREE."""
    filepath = os.path.join(vllm_dir, 'v1', 'core', 'kv_cache_manager.py')
    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} not found")
        return False

    # Backup
    backup = filepath + '.bak'
    if not os.path.exists(backup):
        shutil.copy2(filepath, backup)
        print(f"  Backup: {backup}")

    # 1. Add 'import time'
    add_import_if_missing(filepath, 'import time')

    # 2. BLOCK_ALLOC_FAIL - after the "Cannot allocate" check
    # Anchor: the line that returns None after block allocation failure
    alloc_fail_anchor = "            return None"
    alloc_fail_insert = """
            logger.info(
                "[BLOCK_ALLOC_FAIL] ts=%d req=%s needed=%d free=%d/%d",
                int(time.time() * 1000),
                request.request_id,
                num_blocks_to_allocate,
                self.block_pool.get_num_free_blocks(),
                self.block_pool.num_gpu_blocks - 1,
            )
"""

    # We need to be more specific - find the "Cannot allocate" comment + return None
    with open(filepath, 'r') as f:
        content = f.read()

    # Find "# Cannot allocate" or similar pattern near allocate_slots
    # In 0.17.1, the pattern might be slightly different
    # Look for: if num_blocks > free: return None

    # Strategy: find "return None" that's inside allocate_slots after a free blocks check
    # Use regex to find the block allocation failure pattern

    # Pattern: line with "return None" preceded by a check on free blocks
    fail_pattern = re.compile(
        r'((?:.*(?:free_blocks|num_free|cannot allocate|Cannot allocate).*\n)'
        r'(?:.*\n)*?'  # any lines between
        r'(\s+return None\n))',
        re.IGNORECASE
    )

    # Simpler approach: just find the first "return None" in allocate_slots
    # that follows a check on num_blocks_to_allocate or similar

    # Let's read the file and do line-by-line patching
    with open(filepath, 'r') as f:
        lines = f.readlines()

    patched_lines = []
    in_allocate_slots = False
    alloc_fail_done = False
    alloc_done = False
    free_done = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Track when we enter allocate_slots
        if 'def allocate_slots(' in line:
            in_allocate_slots = True
        elif in_allocate_slots and line.strip().startswith('def ') and 'allocate_slots' not in line:
            in_allocate_slots = False

        # BLOCK_ALLOC_FAIL: before "return None" inside the free blocks check
        if (in_allocate_slots and not alloc_fail_done
            and 'return None' in line.strip()
            and '[BLOCK_ALLOC_FAIL]' not in ''.join(lines[max(0,i-10):i])):
            # Check if there's a free blocks check above
            context = ''.join(lines[max(0,i-5):i])
            if 'free_blocks' in context.lower() or 'num_blocks' in context.lower() or 'cannot allocate' in context.lower() or 'Cannot allocate' in context:
                indent = '            '
                fail_log = f"""{indent}logger.info(
{indent}    "[BLOCK_ALLOC_FAIL] ts=%d req=%s needed=%d free=%d/%d",
{indent}    int(time.time() * 1000),
{indent}    request.request_id,
{indent}    num_blocks_to_allocate,
{indent}    self.block_pool.get_num_free_blocks(),
{indent}    self.block_pool.num_gpu_blocks - 1,
{indent})
"""
                patched_lines.append(fail_log)
                alloc_fail_done = True

        # BLOCK_ALLOC: after allocate_new_blocks returns (look for the assignment)
        if (in_allocate_slots and not alloc_done
            and 'allocate_new_blocks(' in line
            and '[BLOCK_ALLOC]' not in ''.join(lines[i:min(len(lines),i+15)])):
            # Find the end of this call (might span lines)
            patched_lines.append(line)
            i += 1
            # Skip to end of the call (find the closing paren line)
            while i < len(lines) and ')' not in lines[i-1]:
                patched_lines.append(lines[i])
                i += 1

            indent = '        '
            alloc_log = f"""
{indent}num_alloc = sum(len(group) for group in new_blocks)
{indent}if num_alloc > 0:
{indent}    logger.info(
{indent}        "[BLOCK_ALLOC] ts=%d req=%s alloc=%d free=%d/%d",
{indent}        int(time.time() * 1000),
{indent}        request.request_id,
{indent}        num_alloc,
{indent}        self.block_pool.get_num_free_blocks(),
{indent}        self.block_pool.num_gpu_blocks - 1,
{indent}    )
"""
            patched_lines.append(alloc_log)
            alloc_done = True
            continue

        # BLOCK_FREE: wrap the free() method's self.coordinator.free() call
        if ('def free(self, request' in line and not free_done
            and '[BLOCK_FREE]' not in ''.join(lines[i:min(len(lines),i+20)])):
            # Output the def line
            patched_lines.append(line)
            i += 1
            # Find self.coordinator.free( or similar free call
            while i < len(lines):
                fline = lines[i]
                if 'coordinator.free(' in fline or 'self._free_blocks(' in fline:
                    indent = '        '
                    patched_lines.append(f"{indent}free_before = self.block_pool.get_num_free_blocks()\n")
                    patched_lines.append(fline)
                    i += 1
                    # Add the after-free logging
                    free_log = f"""{indent}free_after = self.block_pool.get_num_free_blocks()
{indent}freed = free_after - free_before
{indent}if freed > 0:
{indent}    logger.info(
{indent}        "[BLOCK_FREE] ts=%d req=%s freed=%d free=%d/%d",
{indent}        int(time.time() * 1000),
{indent}        request.request_id,
{indent}        freed,
{indent}        free_after,
{indent}        self.block_pool.num_gpu_blocks - 1,
{indent}    )
"""
                    patched_lines.append(free_log)
                    free_done = True
                    break
                elif fline.strip().startswith('def ') and 'free' not in fline:
                    # We've left the method without finding the free call
                    patched_lines.append(fline)
                    break
                else:
                    patched_lines.append(fline)
                    i += 1
            continue

        patched_lines.append(line)
        i += 1

    # Check if already patched
    content_check = ''.join(patched_lines)
    if '[BLOCK_ALLOC_FAIL]' in content and '[BLOCK_ALLOC]' in content and '[BLOCK_FREE]' in content:
        print("  kv_cache_manager.py: Already patched (all 3 log points present)")
        return True

    with open(filepath, 'w') as f:
        f.writelines(patched_lines)

    results = []
    if alloc_fail_done: results.append("BLOCK_ALLOC_FAIL")
    if alloc_done: results.append("BLOCK_ALLOC")
    if free_done: results.append("BLOCK_FREE")
    print(f"  Added: {', '.join(results) if results else 'NOTHING (check file manually)'}")
    return bool(results)


def patch_scheduler(vllm_dir):
    """Patch scheduler.py with BLOCK_PREEMPT logging."""
    filepath = os.path.join(vllm_dir, 'v1', 'core', 'sched', 'scheduler.py')
    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} not found")
        return False

    # Backup
    backup = filepath + '.bak'
    if not os.path.exists(backup):
        shutil.copy2(filepath, backup)
        print(f"  Backup: {backup}")

    # Add 'import time'
    add_import_if_missing(filepath, 'import time')

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Check if already patched
    content = ''.join(lines)
    if '[BLOCK_PREEMPT]' in content:
        print("  scheduler.py: Already patched (BLOCK_PREEMPT present)")
        return True

    patched_lines = []
    done = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Find _preempt_request method and add logging before kv_cache_manager.free()
        if ('def _preempt_request(' in line and not done):
            patched_lines.append(line)
            i += 1
            # Scan forward for the free() call
            while i < len(lines):
                pline = lines[i]
                if 'kv_cache_manager.free(' in pline and not done:
                    indent = '        '
                    preempt_log = f"""{indent}computed_tokens_lost = request.num_computed_tokens
{indent}logger.info(
{indent}    "[BLOCK_PREEMPT] ts=%d req=%s computed_tokens_lost=%d "
{indent}    "num_preemptions=%d free=%d/%d",
{indent}    int(time.time() * 1000),
{indent}    request.request_id,
{indent}    computed_tokens_lost,
{indent}    request.num_preemptions + 1,
{indent}    self.kv_cache_manager.block_pool.get_num_free_blocks(),
{indent}    self.kv_cache_manager.block_pool.num_gpu_blocks - 1,
{indent})
"""
                    patched_lines.append(preempt_log)
                    patched_lines.append(pline)
                    done = True
                    i += 1
                    break
                elif pline.strip().startswith('def ') and '_preempt' not in pline:
                    patched_lines.append(pline)
                    break
                else:
                    patched_lines.append(pline)
                    i += 1
            continue

        patched_lines.append(line)
        i += 1

    with open(filepath, 'w') as f:
        f.writelines(patched_lines)

    if done:
        print("  Added: BLOCK_PREEMPT")
    else:
        print("  WARNING: Could not find insertion point for BLOCK_PREEMPT")
    return done


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 apply_block_instrumentation.py /path/to/vllm")
        print("Example: python3 apply_block_instrumentation.py /home/ssm-user/venv-vllm/lib64/python3.12/site-packages/vllm")
        sys.exit(1)

    vllm_dir = sys.argv[1]
    if not os.path.isdir(vllm_dir):
        print(f"ERROR: {vllm_dir} is not a directory")
        sys.exit(1)

    print("=== Block Instrumentation Patcher for vLLM 0.17.1 ===\n")

    print("[1/2] Patching kv_cache_manager.py...")
    ok1 = patch_kv_cache_manager(vllm_dir)

    print("\n[2/2] Patching scheduler.py...")
    ok2 = patch_scheduler(vllm_dir)

    print("\n=== Done ===")
    if ok1 and ok2:
        print("All patches applied successfully!")
        print("\nVerify with:")
        print(f"  grep -n 'BLOCK_ALLOC\\|BLOCK_FREE\\|BLOCK_PREEMPT' {vllm_dir}/v1/core/kv_cache_manager.py")
        print(f"  grep -n 'BLOCK_PREEMPT' {vllm_dir}/v1/core/sched/scheduler.py")
    else:
        print("Some patches may need manual attention. Check warnings above.")
        print("Backup files (.bak) were created for safety.")


if __name__ == '__main__':
    main()
