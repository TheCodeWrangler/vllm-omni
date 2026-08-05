# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed event models for tool calling on vLLM-Omni's /v1/realtime endpoint.

These models are the boundary. A `session.update` carrying tools, or a
`conversation.item.create` carrying a tool result, is validated here before it
can reach the prompt renderer or the generation loop, so a malformed payload
comes back as an `error` event with a specific code instead of surfacing later as
a chat-template exception - or as a turn that waits forever for a result whose
`call_id` could never have matched.

Two spellings of the same thing are accepted throughout, because the OpenAI
Realtime API and this endpoint's existing tests spell them differently. The
session fields sit nested under a `session` object

    {"type": "session.update", "session": {"tools": [...], "tool_choice": "auto"}}

or flat on the event; a tool item, and a `tool_choice` naming one function, come
either flat as the Realtime API spells them (`{"type": "function", "name": ...}`)
or wrapped as chat completions does (`{"type": "function", "function": {...}}`).
Everything is normalized to the nested form, which is what the chat template and
the `ChatCompletion...` models reused below expect.

`model` is deliberately not modeled: it is vLLM's own addition to this event,
read straight off the dict by upstream `RealtimeConnection.handle_event`, which
owns both the "Missing required field: model" and "model does not exist" errors.
Upstream reads it off the TOP level, so `RealtimeConnection._with_top_level_model`
lifts a `model` the client nested under `session` before handing the event on -
without validating it, which stays upstream's job.

The tool-call events this endpoint sends back are modeled too, so the wire format
the example client and the integration tests read is declared in one place. The
audio events are not: `response.audio.delta` and `response.audio.done` are still
built as literal dicts at their `send_json` call sites.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ValidationError, model_validator
from pydantic_core import PydanticCustomError
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel

# Client -> server events


def lift_session_fields(data: Any) -> Any:
    """Read the tool fields out of a nested `session` object, if there is one.

    Only a value that is really there overrides the top-level one: an explicit
    `"tools": null` must not destroy a list sent alongside it. Returns a new
    mapping; the caller's event dict is never mutated, because it is handed on to
    upstream's `handle_event` afterwards.

    Module-level and idempotent so a caller can validate the lifted mapping and
    then describe an error against the same mapping pydantic saw
    (`first_error_message`), rather than against a shape one of them normalized
    away.
    """
    if not isinstance(data, dict):
        return data
    session = data.get("session")
    if not isinstance(session, dict):
        return data
    lifted = {key: session[key] for key in ("tools", "tool_choice") if session.get(key) is not None}
    if not lifted:
        return data
    return {**data, **lifted}


def _nest_flat_function(data: Any) -> Any:
    """Move the Realtime API's flat function fields under a `function` key.

    A payload that already nests them - the chat-completions spelling the models
    below are built on - is left as it is.
    """
    if not isinstance(data, dict) or "function" in data:
        return data
    return {
        "type": data.get("type", "function"),
        "function": {key: value for key, value in data.items() if key != "type"},
    }


class OmniTool(ChatCompletionToolsParam):
    """One tool definition, flat or nested."""

    @model_validator(mode="before")
    @classmethod
    def _accept_flat_definition(cls, data: Any) -> Any:
        return _nest_flat_function(data)


class OmniNamedToolChoice(ChatCompletionNamedToolChoiceParam):
    """A `tool_choice` naming one function, flat or nested."""

    @model_validator(mode="before")
    @classmethod
    def _accept_flat_choice(cls, data: Any) -> Any:
        return _nest_flat_function(data)


class OmniSessionUpdate(OpenAIBaseModel):
    """`session.update`, with the tool fields this endpoint adds.

    Named `Omni...` to stay distinguishable from upstream's `SessionUpdate`
    (vllm/entrypoints/speech_to_text/realtime/protocol.py: `type` and `model`).
    """

    type: Literal["session.update"] = "session.update"
    # `None` means "this event does not carry the field", which leaves the
    # session's current value alone - the way a `session.update` without `tools`
    # has always been treated. `tool_choice` deliberately does not default to
    # `"auto"`, so an update that changes only something else cannot silently
    # reset a choice the client set earlier.
    tools: list[OmniTool] | None = None
    tool_choice: Literal["none", "auto", "required"] | OmniNamedToolChoice | None = None

    @model_validator(mode="before")
    @classmethod
    def _lift_session_fields(cls, data: Any) -> Any:
        """Accept the nested `session` spelling here too, so the model is usable
        on a raw event and not only on an already-lifted mapping."""
        return lift_session_fields(data)


class FunctionCallOutputItem(OpenAIBaseModel):
    """The `item` of a `conversation.item.create`: one tool's result.

    `handle_event` owns which `item.type` values this endpoint routes at all (it
    refuses anything else with `unsupported_item`), so `type` is fixed here rather
    than modeled as a union.
    """

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str

    @model_validator(mode="before")
    @classmethod
    def _reject_malformed_result(cls, data: Any) -> Any:
        """Report the two shape errors in this endpoint's own words.

        The field types above already reject both, but pydantic's generic "Field
        required" does not name the event, nor say that an empty `call_id` is no
        better than a missing one. These messages go straight to the client; the
        error type is the code they are reported under.
        """
        if not isinstance(data, dict):
            return data
        call_id = data.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise PydanticCustomError(
                "invalid_function_call_output",
                "function_call_output requires a non-empty string 'call_id'",
            )
        output = data.get("output")
        if not isinstance(output, str):
            raise PydanticCustomError(
                "invalid_function_call_output",
                "function_call_output 'output' must be a string, got {output_type}",
                {"output_type": type(output).__name__},
            )
        return data


# Server -> client events


class FunctionCallItem(OpenAIBaseModel):
    """The `item` of `response.output_item.added`: a tool call the model started."""

    type: Literal["function_call"] = "function_call"
    name: str
    call_id: str


class ResponseOutputItemAdded(OpenAIBaseModel):
    """The model has started a tool call; its name and `call_id` are now known."""

    type: Literal["response.output_item.added"] = "response.output_item.added"
    item: FunctionCallItem


class ResponseFunctionCallArgumentsDelta(OpenAIBaseModel):
    """More of a tool call's `arguments` JSON text, as it is generated."""

    type: Literal["response.function_call_arguments.delta"] = "response.function_call_arguments.delta"
    call_id: str
    delta: str


class ResponseFunctionCallArgumentsDone(OpenAIBaseModel):
    """A tool call is complete: run it and answer with `conversation.item.create`."""

    type: Literal["response.function_call_arguments.done"] = "response.function_call_arguments.done"
    call_id: str
    arguments: str


def _child(node: Any, key: Any) -> Any:
    """The part of `node` that `key` selects, or None if there is none."""
    if isinstance(node, dict):
        return node.get(key)
    if isinstance(node, list) and isinstance(key, int) and -len(node) <= key < len(node):
        return node[key]
    return None


def _describe_location(loc: tuple[Any, ...], payload: Any) -> str:
    """`loc` as a dotted field path, in the terms the client's own payload uses.

    Two things pydantic puts in a path have no meaning to the client: the segment
    a union contributes per candidate type (`literal['none',...]`,
    `function-wrap[...]`), and the `function` level this module *adds* to a flat
    tool item. Naming the latter told a client that sent
    `{"type": "function", "name": ...}` to fix a `function` key it never wrote,
    which is why `payload` is walked alongside `loc`.
    """
    parts: list[str] = []
    node = payload
    for part in loc:
        if isinstance(part, str) and "[" in part:
            continue
        if part == "function" and isinstance(node, dict) and "function" not in node:
            continue  # `_nest_flat_function` added this level, the client did not
        parts.append(str(part))
        node = _child(node, part)
    return ".".join(parts)


def first_error_message(exc: ValidationError, payload: Any = None) -> str:
    """The first problem in `exc` as a single line, for an `error` event.

    Pydantic's own `str(exc)` is multi-line and carries a model name and a docs
    URL, which reads badly on the wire; only the first problem is reported, since
    the client has to fix it and resend either way. The field path is prefixed when
    there is one; pass the `payload` that was validated to have it described in the
    shape the client sent (see `_describe_location`). The class name pydantic names
    in "or instance of OmniTool" goes the same way: these models are this module's
    business, not the client's. The messages raised above report at the model level,
    so they come through verbatim.
    """
    error = exc.errors()[0]
    location = _describe_location(error["loc"], payload)
    message = str(error["msg"]).split(" or instance of ")[0]
    return f"{location}: {message}" if location else message
