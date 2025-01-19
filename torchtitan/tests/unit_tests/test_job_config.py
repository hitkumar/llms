import tempfile

import pytest
import tomli_w
from config_manager import JobConfig


class TestJobConfig:
    def test_job_config_file(self):
        config = JobConfig()
        config.parse_args(["--job.config_file", "./train_configs/debug_model.toml"])
        assert config.training.steps == 10
        assert config.training.seq_len == 2048
        assert config.metrics.log_freq == 1

    def test_job_file_does_not_exist(self):
        with pytest.raises(FileNotFoundError):
            config = JobConfig()
            config.parse_args(["--job.config_file", "ohno.toml"])
