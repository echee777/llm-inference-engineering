# Experiment Value Filter

Decision framework for whether a syllabus experiment should be run or replaced with conceptual coverage. Applied per-experiment, not per-day.

---

## The Tradeoff

GPU hours and study time are finite. Every hour spent confirming a predictable outcome is an hour not spent learning a concept that could differentiate in an interview at a frontier AI company. The goal is not to skip experiments. The goal is to run only experiments that teach something that reasoning alone cannot.

---

## Run the Experiment If ANY of These Are True

1. The outcome is non-obvious or non-linear
   - The experiment could reveal behavior that cannot be predicted from first principles or prior data
   - Example: the Day 24 KV utilization cliff. The existence of a cliff is predictable but its exact location (87%), sharpness (2.4x p99 jump over 1.7pp), and the recompute fraction (30%) are not deducible without measurement

2. The experiment builds a skill you'd use in production
   - Writing the instrumentation, running the sweep, interpreting live metrics
   - The act of doing it creates muscle memory for on-call debugging or capacity planning
   - Example: writing a Prometheus query to detect divergence ratio in real time

3. The mechanism is disputed or counterintuitive
   - Reasonable engineers would disagree about what happens
   - Example: "does chunked prefill reduce throughput?" (answer: no at low utilization, yes at high -- not obvious which regime you're in)

4. The experiment produces a specific number needed downstream
   - A later deliverable, design doc, or postmortem requires a measured value, not a range estimate
   - Example: Day 24 cliff point (87%) feeds directly into admission-control-retrofit.md threshold (85%)

5. You cannot yet explain the full causal chain from memory
   - If you'd struggle to whiteboard the mechanism in an interview without having seen it, run it
   - The experiment cements understanding that conceptual coverage alone didn't achieve

## Skip the Experiment If ALL of These Are True

1. The outcome is fully deducible from prior data + known mechanisms
   - You can write down what will happen, why, and the approximate magnitude before running it
   - Example: retry amplification follows a geometric series. If you know the timeout rate and max attempts, you know the amplification factor

2. The concept is well-established in industry
   - Exponential backoff, circuit breakers, retry storms are documented extensively (Google SRE book, AWS architecture blog, etc.)
   - No frontier AI company will be impressed that you confirmed retries cause load amplification

3. The measured numbers are not reusable
   - The exact values are workload-specific, hardware-specific, and config-specific
   - They won't transfer to a different model, different GPU, or different traffic pattern
   - Example: "retry storm peak amplification was 1.73" is not a transferable finding

4. The time can be redirected to a higher-value concept
   - There exists a learnable concept (in the current syllabus or adjacent topics) that is more differentiating for employment at frontier AI companies
   - The replacement concept should be identifiable at decision time, not hypothetical

---

## Application to Syllabus Design

When writing a day's syllabus, apply this filter to each proposed experiment:

- Tag experiments as RUN (meets at least one "run" criterion) or CONCEPTUAL (meets all "skip" criteria)
- For CONCEPTUAL experiments: replace GPU time with deeper conceptual coverage, interview-grade Q&A, or coverage of adjacent high-value topics
- For RUN experiments: protect the full time allocation. Do not compress experiments that meet the "run" criteria
- When in doubt, bias toward running. A false skip costs more than a false run because gaps in hands-on experience are harder to backfill than gaps in conceptual knowledge

---

## Examples

```
Experiment                                    Verdict     Reasoning
Day 24 KV cliff sweep                        RUN         Non-linear outcome, exact cliff location
                                                          needed for admission control design,
                                                          cannot be deduced from prior data

Day 23 chunked prefill interference           RUN         Counterintuitive (throughput cost is zero
                                                          at low util), produces reusable comparison
                                                          numbers, builds profiling skill

Day 26 retry storm induction                  CONCEPTUAL  Outcome fully deducible (geometric series),
                                                          well-established industry concept, measured
                                                          numbers not transferable

Day 26 backoff variant comparison             CONCEPTUAL  Direction known (backoff reduces amplification),
                                                          exact delta is workload-specific, no downstream
                                                          deliverable requires the number
```
