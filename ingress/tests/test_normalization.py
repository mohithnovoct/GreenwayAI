import hashlib

from ingress.normalization import build_gateway_request, canonicalize, normalize_payload, prompt_text


def test_multiline_content_is_preserved_but_hash_is_canonicalized():
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "  first line  \nsecond\tline\n\nthird  "}],
    }

    request = build_gateway_request(payload, None)
    assert request.raw_payload["messages"][0]["content"] == "first line \nsecond line\n\nthird"
    expected_prompt = "user: first line second line third"
    assert request.normalized_prompt == expected_prompt
    assert request.content_hash == hashlib.sha256(expected_prompt.encode()).hexdigest()


def test_equivalent_whitespace_has_same_hash():
    first = {"model": "test-model", "messages": [{"role": "user", "content": "hello\n\nworld"}]}
    second = {"model": "test-model", "messages": [{"role": "user", "content": " hello   world "}]}

    assert build_gateway_request(first, None).content_hash == build_gateway_request(second, None).content_hash


def test_code_and_json_structure_survive_sanitization():
    content = """  ```json
{  "name":  "Greenway" }
```
  """
    normalized = normalize_payload({"messages": [{"role": "user", "content": content}]})

    assert normalized["messages"][0]["content"] == '```json\n{ "name": "Greenway" }\n```'


def test_multimodal_text_parts_are_sanitized_recursively():
    first = {
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": "  hello   world  \n\nfoo"}, {"type": "image_url", "image_url": {"url": "..."}}],
        }],
    }
    second = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hello world\n\nfoo"}, {"type": "image_url", "image_url": {"url": "..."}}]}]}
    normalized = normalize_payload(first)

    assert normalized["messages"][0]["content"][0]["text"] == "hello world \n\nfoo"
    assert normalized["messages"][0]["content"][1] == {"type": "image_url", "image_url": {"url": "..."}}
    assert build_gateway_request(first, None).content_hash == build_gateway_request(second, None).content_hash