# Signal Game Equilibrium Algorithm for LLM Sampling
This is a mock reproduction of: [The Consensus Game: Language Model Generation via Equilibrium Search](http://arxiv.org/abs/2310.09139)

Please refer to the original work for a dissection on algorithm.

I used **MMLU Benchmark** on `gpt-2` in a **next-token prediction** fashion. (Note that this is not deterministic)

Ideally, two models will finally decide on one of the given options after 5000 iterations.

Terminal will resemble:

```
Random Correctness: correct
最终生成器策略：｛'0'：1.8468149259989305e-10，'4'：1.7767062464098044e-10，'2'： 0.9999999994601962，'6'：1.7745170128845816e-10｝
最终判别器策略：｛'0'：1.8468149646498393e-10，'4'：1.776706026385534be-10，'2'： 0.9999999994601962.'6'：1.7745166738332201e-10｝
```

Enjoy "gaming"!
