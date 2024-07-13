import fire
from finetune.args import TrainArgs

def train(config: str):
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
    print(f"args: {args}")

if __name__ == "__main__":
    fire.Fire(train)
    