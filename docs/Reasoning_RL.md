**Papers**
- Deepseek-R1: https://arxiv.org/pdf/2501.12948
Very nice paper, distillation is a very neat idea.

**Articles**
- https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training
   - Introduces RLHF and motivates RLVR
   - Goes into some recent papers like below post.
   - Best to play with these in code.
- https://www.interconnects.ai/p/papers-im-reading-base-model-rl-grpo

**Code**
- try in Torchtune
- TinyZero: https://github.com/Jiayi-Pan/TinyZero
- nano aha moment: https://github.com/McGill-NLP/nano-aha-moment


**Videos**

vLLM intro: https://www.youtube.com/watch?v=9ih0EmcXRHE&t=30s

* Deepseek-R1
    * Umar Jamil: https://www.youtube.com/watch?v=XMnxKGVnEUc
    * HuggingFace: https://www.youtube.com/watch?v=1xDVbu-WaFo

GRPO new variants and implementation secrets:
https://www.youtube.com/watch?v=amrJDwMUFNs

CS 285 UC berkeley: https://rail.eecs.berkeley.edu/deeprlcourse/
- Finished lecture 2 and 4
- Lecture 5 on policy gradients done.
- Math is still not srtting in fully, might need to go through it again.
- Kevin Murphy's book might be useful to go through.
- Need see code as well (TorchRL could be a good starting point)
- Emma's stanford lectures could be another useful resource.

One way to look at policy gradients loss is
pg_loss = -advantages * ratio of policy_new / policy_old
(page 62 od RLHF book). Everything else is just improving model stability from there.

Stanford lecture from DPO authors: https://www.youtube.com/watch?v=Q7rl8ovBWwQ&list=PLoROMvodv4rN4wG6Nk6sNpTEbuOSosZdX&index=9
- Good intro to DPO, shares some intuition with PPO at the end with stating it is a weaker optimization method which leads to some regularization causing improved performance.
- Nice formulation of RLHF objective, slides: https://web.stanford.edu/class/cs234/CS234Spr2024/slides/dpo_slides.pdf
