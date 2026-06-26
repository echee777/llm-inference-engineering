Let me try this without your help

Ridge point Calculation
- T4 tesla - 320GB/s memory bw. 65 TFlops => 65000/320 or 203 Flops/B

AI calculation
- Given a 7B model FP16 => weights = 14GBytes
- Per decode step 2FLops/PARAMETER => 7E9*2 = 14GFlops
- AI for decode = 14E9/14E9 = 1 Flops/Byte
- AI for prefill of length 2048 tokens assuming 2Flop/PARAMETER  = 2*7E9 -= 14GFlop/token => 14 * 2048 GFlop .  Hence AI = 2048 * 14GFlop / 14GB = 2048 Flops/Byte

Given the T4
- the ridge point is 203 Flops/B
- decode is 1Flop/B which lies far to the left of the ridge point and firmly in memory bound regime
- prefill is at 2048 Flop/B which is far to the right of the ridge point and firmly in compute bound regime
- the intuition is that decode requires re-reading of the weights from memory for each output token hence firmly memory bound
- the intuition for prefill is that each token requires matmul over all the weights but a single more-or-less shared read of the weights from memory, hence firmly compute bound

ChatGPT: On a T4 (65 TFLOPs FP16, 320 GB/s bandwidth), the ridge point is approximately 203 FLOPs/byte. A 7B FP16 model has ~14 GB of weights and requires ~14 GFLOPs per decode step, giving an arithmetic intensity of ~1 FLOP/byte. Since 1 ≪ 203, decode is deeply memory-bandwidth-bound and can achieve at most ~0.5% of peak compute. In contrast, prefill with a 2048-token prompt performs ~28.7 TFLOPs while reading the same ~14 GB of weights once, yielding ~2048 FLOPs/byte. This lies far to the right of the ridge point and is compute-bound. The asymmetry arises because decode re-streams the model weights for every generated token, whereas prefill amortizes a single weight stream across many tokens.