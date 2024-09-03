import pytest
import asyncio
from unittest.mock import patch, MagicMock
from clicking.common.image_utils import ImageProcessorBase
from pydantic import BaseModel, Field

class TestOutputType(BaseModel):
    name: str = Field(..., max_length=50)
    description: str
    value: int

class TestImageProcessorBase:
    @pytest.fixture
    def image_processor(self):
        return ImageProcessorBase()

    @pytest.mark.asyncio
    async def test_get_image_response_valid_output(self, image_processor):
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": '{"name": "Test", "description": "A test object", "value": 42}'
                    }
                }
            ]
        }

        with patch('clicking.common.image_utils.acompletion', new_callable=MagicMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            result = await image_processor._get_image_response(
                "base64_image_data",
                "text_prompt",
                [],
                TestOutputType
            )

        assert isinstance(result, TestOutputType)
        assert result.name == "Test"
        assert result.description == "A test object"
        assert result.value == 42

    @pytest.mark.asyncio
    async def test_get_image_response_invalid_json(self, image_processor):
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": 'Invalid JSON'
                    }
                }
            ]
        }

        with patch('clicking.common.image_utils.acompletion', new_callable=MagicMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            with pytest.raises(ValueError, match="Response does not match the specified output type"):
                await image_processor._get_image_response(
                    "base64_image_data",
                    "text_prompt",
                    [],
                    TestOutputType
                )

    @pytest.mark.asyncio
    async def test_get_image_response_missing_field(self, image_processor):
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": '{"name": "Test", "description": "A test object"}'
                    }
                }
            ]
        }

        with patch('clicking.common.image_utils.acompletion', new_callable=MagicMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            with pytest.raises(ValueError, match="Response does not match the specified output type"):
                await image_processor._get_image_response(
                    "base64_image_data",
                    "text_prompt",
                    [],
                    TestOutputType
                )

    @pytest.mark.asyncio
    async def test_get_image_response_invalid_field_type(self, image_processor):
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": '{"name": "Test", "description": "A test object", "value": "not an integer"}'
                    }
                }
            ]
        }

        with patch('clicking.common.image_utils.acompletion', new_callable=MagicMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            with pytest.raises(ValueError, match="Response does not match the specified output type"):
                await image_processor._get_image_response(
                    "base64_image_data",
                    "text_prompt",
                    [],
                    TestOutputType
                )

if __name__ == '__main__':
    pytest.main()