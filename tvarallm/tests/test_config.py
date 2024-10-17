from pathlib import Path

import pytest

from tvarallm.config import Config


def test_config_from_yaml():
    # Create a temporary YAML file
    yaml_content = """
    name: test_model
    vocab_size: 50000
    n_layer: 12
    n_head: 16
    n_embd: 768
    """
    tmp_path = Path("test_config.yaml")
    tmp_path.write_text(yaml_content)

    # Load the config from the YAML file
    config = Config.from_yaml(tmp_path)

    # Check if the values are correctly loaded
    assert config.name == "test_model"
    assert config.vocab_size == 50000
    assert config.n_layer == 12
    assert config.n_head == 16
    assert config.n_embd == 768

    # Clean up the temporary file
    tmp_path.unlink()
