import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# --- Data from ramp5 (memory-constrained, gpu_mem=0.45) ---
r5_conc =    [1,    2,    4,    6,    8,   10,   12,   16,   20,   24,   32]
r5_kv =      [14.0, 29.2, 60.0, 92.5, 94.4, 95.4, 94.4, 95.4, 95.4, 95.2, 95.4]
r5_queue =   [0,    0,    0,    0,    1,    3,    5,    9,   13,   17,   25]
r5_run =     [0,    1,    3,    5,    6,    6,    6,    6,    6,    6,    6]
r5_p50 =     [581.3, 1106.6, 1702.6, 1971.8, 2507.7, 2508.0, 2510.7, 2553.5, 2550.5, 2588.7, 2598.4]
r5_p99 =     [582.4, 1172.9, 2254.8, 3303.6, 3494.0, 3488.4, 3492.7, 3535.9, 3532.5, 3571.5, 3584.9]
r5_preempt = [0,    0,    0,   48,  113,  178,  243,  308,  373,  437,  502]
r5_toks =    [33.8, 67.6, 101.6, 101.2, 50.7, 50.7, 50.6, 50.8, 50.7, 50.8, 50.7]

# --- Data from ramp3 (compute-constrained, gpu_mem=0.90) ---
r3_conc =    [1,    2,    4,    6,    8,   10,   12,   16,   20,   24,   32]
r3_kv =      [13.8, 27.9, 55.0, 80.9, 96.3, 96.4, 94.8, 90.2, 96.5, 92.5, 96.2]
r3_queue =   [0,    0,    0,    0,    0,    2,    4,    8,   12,   16,   24]
r3_run =     [1,    2,    4,    5,    6,    6,    6,    6,    6,    6,    6]
r3_p50 =     [599.3, 635.1, 637.5, 642.7, 1433.2, 3040.8, 4808.3, 8051.0, 11403.6, 14441.7, 20950.2]
r3_p99 =     [604.5, 1161.2, 1148.5, 1974.4, 5997.2, 6860.4, 8775.8, 13332.2, 14625.2, 17707.9, 24550.1]
r3_preempt = [0]*11
r3_toks =    [28.1, 40.0, 61.5, 69.2, 74.6, 73.5, 77.1, 72.6, 77.1, 73.1, 75.8]

fig, axes = plt.subplots(3, 2, figsize=(14, 14))
fig.suptitle("Postmortem #1: Ramp-to-Failure Analysis\nMemory-constrained (0.45) vs Compute-constrained (0.90)",
             fontsize=13, fontweight='bold', y=0.98)

# --- Graph 1: KV cache utilization % vs concurrency ---
ax = axes[0, 0]
ax.plot(r5_conc, r5_kv, 'o-', color='#d62728', label='Memory-constrained (0.45)', linewidth=2, markersize=6)
ax.plot(r3_conc, r3_kv, 's-', color='#1f77b4', label='Compute-constrained (0.90)', linewidth=2, markersize=6)
ax.axhline(y=92.5, color='#d62728', linestyle='--', alpha=0.5, label='Preemption onset (92.5%)')
ax.set_xlabel('Concurrency')
ax.set_ylabel('KV Cache Utilization (%)')
ax.set_title('1. KV Cache Utilization vs Concurrency')
ax.legend(fontsize=8)
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3)

# --- Graph 2: TTFT p50 and p99 vs concurrency ---
ax = axes[0, 1]
ax.plot(r5_conc, [x/1000 for x in r5_p50], 'o-', color='#d62728', label='Mem p50', linewidth=1.5, markersize=5)
ax.plot(r5_conc, [x/1000 for x in r5_p99], 'o--', color='#d62728', label='Mem p99', linewidth=1.5, markersize=5, alpha=0.7)
ax.plot(r3_conc, [x/1000 for x in r3_p50], 's-', color='#1f77b4', label='Compute p50', linewidth=1.5, markersize=5)
ax.plot(r3_conc, [x/1000 for x in r3_p99], 's--', color='#1f77b4', label='Compute p99', linewidth=1.5, markersize=5, alpha=0.7)
ax.set_xlabel('Concurrency')
ax.set_ylabel('TTFT (seconds)')
ax.set_title('2. TTFT p50 and p99 vs Concurrency')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Graph 3: Preemption count vs concurrency ---
ax = axes[1, 0]
ax.plot(r5_conc, r5_preempt, 'o-', color='#d62728', label='Memory-constrained (0.45)', linewidth=2, markersize=6)
ax.plot(r3_conc, r3_preempt, 's-', color='#1f77b4', label='Compute-constrained (0.90)', linewidth=2, markersize=6)
ax.set_xlabel('Concurrency')
ax.set_ylabel('Cumulative Preemptions')
ax.set_title('3. Preemption Events vs Concurrency')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Graph 4: Concurrency vs KV utilization (scatter) ---
ax = axes[1, 1]
ax.scatter(r5_kv, r5_conc, color='#d62728', s=80, zorder=5, label='Memory-constrained (0.45)', edgecolors='black', linewidth=0.5)
ax.scatter(r3_kv, r3_conc, color='#1f77b4', s=80, zorder=5, label='Compute-constrained (0.90)', marker='s', edgecolors='black', linewidth=0.5)
for i, c in enumerate(r5_conc):
    ax.annotate(f'c={c}', (r5_kv[i], r5_conc[i]), textcoords="offset points", xytext=(8, 0), fontsize=7, color='#d62728')
for i, c in enumerate(r3_conc):
    ax.annotate(f'c={c}', (r3_kv[i], r3_conc[i]), textcoords="offset points", xytext=(8, 0), fontsize=7, color='#1f77b4')
ax.set_xlabel('KV Cache Utilization (%)')
ax.set_ylabel('Concurrency')
ax.set_title('4. Concurrency vs KV Utilization (Scatter)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Graph 5: Queue depth vs concurrency ---
ax = axes[2, 0]
ax.plot(r5_conc, r5_queue, 'o-', color='#d62728', label='Memory-constrained (0.45)', linewidth=2, markersize=6)
ax.plot(r3_conc, r3_queue, 's-', color='#1f77b4', label='Compute-constrained (0.90)', linewidth=2, markersize=6)
ax.set_xlabel('Concurrency')
ax.set_ylabel('Queue Depth (num_waiting_seqs)')
ax.set_title('5. Queue Depth vs Concurrency')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Bonus: Throughput vs concurrency ---
ax = axes[2, 1]
ax.plot(r5_conc, r5_toks, 'o-', color='#d62728', label='Memory-constrained (0.45)', linewidth=2, markersize=6)
ax.plot(r3_conc, r3_toks, 's-', color='#1f77b4', label='Compute-constrained (0.90)', linewidth=2, markersize=6)
ax.set_xlabel('Concurrency')
ax.set_ylabel('Throughput (tok/s)')
ax.set_title('Bonus: Throughput vs Concurrency')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/home/ssm-user/phase-b/day22/postmortem_graphs.png', dpi=150, bbox_inches='tight')
print("Saved to day22/postmortem_graphs.png")
