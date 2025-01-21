**Courses/Talks**

***CMU Multimodal AI***
- https://cmu-multicomp-lab.github.io/mmml-course/fall2023/schedule/
- Lecture 1 and 2 done
- Lecture 3
Talks about unimodal resprsentation for images. CNNs mostly.

Coding Paligemma from scratch - https://www.youtube.com/watch?v=vAmKB7iPkWw&t=2138s

Talk from Lucas Beyer - https://www.youtube.com/watch?v=kxO6ARgI_SU&t=2621s

EEML Talk - https://www.youtube.com/watch?v=rUQUv4u7jFs

Molmo vs Llama 3.2 vision - https://www.youtube.com/watch?v=06sKEzOvop8
- Molmo is a better vision model, while Llama 3.2 is a better text model.
- Pixmo could be very interesting to look into.
- Multimodal benchmarks are missing.

Latent Space talk on Molmo - https://www.youtube.com/watch?v=8BN9CdIYaqc

Distributed Training Berkeley - https://www.youtube.com/watch?v=9TwTKt50ZG8
Picotron Part 1 and Part 2: https://www.youtube.com/watch?v=qUMPaSWi5HI&list=PL-_armZiJvAnhcRr6yTJ0__f3Oi-LLi9S&index=2
- Nice intro to tensor parallel.
Torchtian talk - https://www.youtube.com/watch?v=VYWRjcUqW6w&t=2002s
FSDP - https://www.youtube.com/watch?v=By_O0k102PY
From torchtian paper: "here TP shards within nodes and FSDP shards across nodes"

**Papers**
- Paligemma
- Paligemma 2
- Torchtitan paper
- AnyMal paper - https://aclanthology.org/2024.emnlp-industry.98.pdf
- Molmo paper - https://arxiv.org/pdf/2409.17146
    - Great paper, lot of great ideas
    - Captions with length hints
    - Datasets
    - Evaluation
    - Train with Torchtitan

To Read

- OlMoe
    - https://www.youtube.com/watch?v=3bG7hqTMAaQ
- Flamingo
- Deepseek v3

**Code**
- Paligemma inference code: https://github.com/hitkumar/llms/tree/main/paligemma-pytorch
- Implement TP and FSDP for llama 3.1 model.
- is FSDP alone enough to train Llama 3 model on H100?

To go through
- Picotron
