# Abstract — Version 2

We investigate memory-augmented adaptation for a local code-agent based on
Qwen2.5-Coder-1.5B-Instruct fine-tuned with LoRA. The central finding is that
verified offline memory retrieval is load-bearing: removing the 99-record index
drops performance by 17.8 percentage points on a frozen 28-task benchmark
(82.1% → 64.3%). Five follow-up experiments test whether this baseline can be
improved. Continued LoRA training regresses at every tested configuration, by
17.9–25.0 pp, due to distribution shift in output format. Targeted repair memory
fixes four known failures diagnostically (96.4% on the same frozen benchmark),
but this result is not a clean held-out champion: the benchmark is no longer
independent because repair records target known failures. On a 32-task clean
generalisation benchmark with zero overlap, the original champion index (62.5%)
outperforms the repair-enhanced index (56.2%). Three routing strategies — family
routing, TF-IDF confidence routing, and oracle analysis — confirm that no
deployable strategy improves over champion on clean tasks. The oracle ceiling is
71.9%, with 9 of 32 tasks failing regardless of index selection. We identify
three retrieval failure modes rooted in the same cause: TF-IDF similarity measures
vocabulary co-occurrence rather than algorithmic relevance. The clean champion
(23/28 = 82.1%) is fully reproducible on a single consumer GPU.
