import pytest
from clicking.prompt_manager.core import PromptManager
from unittest.mock import mock_open, patch
import os

@pytest.fixture
def sample_prompt_content():
    file_path = "./test/prompts/prompt_refiner.md"
    with open(file_path, 'r') as file:
        return file.read()

@pytest.fixture
def prompt_manager(tmp_path):
    file_path = tmp_path / "prompts.txt"
    return PromptManager(str(file_path))

def test_load_prompts(prompt_manager, sample_prompt_content):
    with patch("builtins.open", mock_open(read_data=sample_prompt_content)):
        prompts = prompt_manager.load_prompts()
    
    assert prompts["system_prompt"].startswith("You are an AI assistant designed to help refine")
    assert "Refine" in prompts["user_prompts"]
    assert "Analyze" in prompts["user_prompts"]
    assert "Generate" in prompts["user_prompts"]

def test_parse_user_prompts(prompt_manager, sample_prompt_content):
    user_prompt_content = sample_prompt_content.split("# User prompt")[1]
    parsed_prompts = prompt_manager.parse_user_prompts(user_prompt_content)
    
    assert "Refine" in parsed_prompts
    assert "Analyze" in parsed_prompts
    assert "Generate" in parsed_prompts

def test_get_prompt_system(prompt_manager, sample_prompt_content):
    with patch("builtins.open", mock_open(read_data=sample_prompt_content)):
        system_prompt = prompt_manager.get_prompt(type="system")
    
    assert system_prompt.startswith("You are an AI assistant designed to help refine")

def test_get_prompt_user(prompt_manager, sample_prompt_content):
    with patch("builtins.open", mock_open(read_data=sample_prompt_content)):
        user_prompt = prompt_manager.get_prompt(type="user", prompt_key="Refine", template_values={"original_prompt": "Test prompt"})
    
    assert "Test prompt" in user_prompt
    assert "Clarity:" in user_prompt

def test_get_prompt_invalid_type(prompt_manager):
    with patch.object(PromptManager, 'load_prompts', return_value={'system_prompt': '', 'user_prompts': {}}):
        with pytest.raises(ValueError, match="Prompt type 'invalid' not found in prompt dictionary."):
            prompt_manager.get_prompt(type="invalid")

def test_get_prompt_missing_key(prompt_manager):
    with patch.object(PromptManager, 'load_prompts', return_value={'system_prompt': '', 'user_prompts': {}}):
        with pytest.raises(ValueError, match="Prompt key is required for user prompts."):
            prompt_manager.get_prompt(type="user")

def test_get_prompt_invalid_key(prompt_manager, sample_prompt_content):
    with patch("builtins.open", mock_open(read_data=sample_prompt_content)):
        with pytest.raises(ValueError, match="User prompt 'InvalidKey' not found in prompt dictionary."):
            prompt_manager.get_prompt(type="user", prompt_key="InvalidKey")

def test_duplicate_user_prompt_heading(prompt_manager):
    duplicate_content = """
## Greeting
Hello!

## Greeting
Duplicate greeting!
"""
    with pytest.raises(ValueError, match="Duplicate subheading 'Greeting' found."):
        prompt_manager.parse_user_prompts(duplicate_content)

def test_invalid_heading(prompt_manager):
    invalid_content = """
# Invalid heading
This is an invalid heading.
"""
    with patch("builtins.open", mock_open(read_data=invalid_content)):
        with pytest.raises(ValueError, match="Heading 'Invalid heading' not found in prompt dictionary."):
            prompt_manager.load_prompts()

# Test descriptions:

# • test_get_prompt_system: Verifies that the system prompt is correctly retrieved and starts with the expected text.
# • test_get_prompt_user: Ensures that user prompts are correctly retrieved and formatted with template values.
# • test_get_prompt_invalid_type: Checks that an appropriate error is raised when an invalid prompt type is requested.
# • test_get_prompt_missing_key: Verifies that an error is raised when a key is not provided for user prompts.
# • test_get_prompt_invalid_key: Ensures that an appropriate error is raised when an invalid user prompt key is used.
# • test_duplicate_user_prompt_heading: Checks that an error is raised when duplicate user prompt headings are found.
# • test_invalid_heading: Verifies that an error is raised when an invalid heading is encountered in the prompt file.
