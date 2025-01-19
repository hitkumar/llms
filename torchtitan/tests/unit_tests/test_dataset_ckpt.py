import torch

from core.datasets import build_hf_data_loader, build_tokenizer


class TestDatasetCheckpointing:
    def test_training_resumption(self):
        print("test_training_resumption")
        dataset_name = "c4_test"
        dataset_path = "./tests/assets/c4_test"
        dl = self._build_dataloader(dataset_name, dataset_path)
        it = iter(dl)
        next(it)

        state = dl.state_dict()
        expected_input, expected_target = next(it)

        dl2 = self._build_dataloader(dataset_name, dataset_path)
        dl2.load_state_dict(state)
        input, target = next(iter(dl2))

        assert torch.equal(input, expected_input)
        assert torch.equal(target, expected_target)

    def _build_dataloader(self, dataset_name, dataset_path):
        tokenizer = build_tokenizer("tiktoken", "./tests/assets/test_tiktoken.model")
        return build_hf_data_loader(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            batch_size=1,
            seq_len=1024,
            world_size=4,
            rank=0,
        )


if __name__ == "__main__":
    test_dataset_checkpointing = TestDatasetCheckpointing()
    test_dataset_checkpointing.test_training_resumption()
