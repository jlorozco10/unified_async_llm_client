import pytest
from pydantic import ValidationError
from schemas import ChatMessage

class TestChatMessage:
    @pytest.mark.parametrize("role",["system", "user", "assistant"])
    def test_accepts_valid_roles(self, role):
        msg = ChatMessage(role = role, content="Hello")
        assert msg.role == role

    def test_rejects_unknown_role(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="bot", content="Hello")

# ToDo:
# TestChatMessage
# - empty content is rejected
# - whitespace-only content is rejected ("   ")

# TestModelConfig
# - only model given → check the defaults for temperature and max_tokens
# - temperature 0.0 and 2.0 are accepted (parametrize)
# - temperature -0.1 and 2.1 are rejected (parametrize)
# - max_tokens of 0 and -1 are rejected
# - missing model is rejected

# TestModelResponse
# - the five ok cases from the table in my last message
# - model_name is required