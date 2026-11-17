import json
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, cast

from rich.live_render import VerticalOverflowMethod

from ..cache import Cache
from ..config import cfg
from ..function import get_function
from ..printer import MarkdownPrinter, Printer, TextPrinter
from ..role import DefaultRoles, SystemRole

completion: Callable[..., Any] = lambda *args, **kwargs: Generator[Any, None, None]

base_url = cfg.get("API_BASE_URL")
use_litellm = cfg.get("USE_LITELLM") == "true"
additional_kwargs = {
    "timeout": int(cfg.get("REQUEST_TIMEOUT")),
    "api_key": cfg.get("OPENAI_API_KEY"),
    "base_url": None if base_url == "default" else base_url,
}

if use_litellm:
    import litellm  # type: ignore

    completion = litellm.completion
    litellm.suppress_debug_info = True
else:
    from openai import OpenAI

    client = OpenAI(**additional_kwargs)  # type: ignore
    completion = client.chat.completions.create
    additional_kwargs = {}


class Handler:
    cache = Cache(int(cfg.get("CACHE_LENGTH")), Path(cfg.get("CACHE_PATH")))

    def __init__(self, role: SystemRole, markdown: bool) -> None:
        self.role = role

        api_base_url = cfg.get("API_BASE_URL")
        self.base_url = None if api_base_url == "default" else api_base_url
        self.timeout = int(cfg.get("REQUEST_TIMEOUT"))

        self.markdown = "APPLY MARKDOWN" in self.role.role and markdown
        self.code_theme, self.color = cfg.get("CODE_THEME"), cfg.get("DEFAULT_COLOR")

    @property
    def printer(self) -> Printer:
        vertical_overflow = cast(
            VerticalOverflowMethod, cfg.get("MARKDOWN_LIVE_VERTICAL_OVERFLOW")
        )
        refresh_interval = float(cfg.get("MARKDOWN_LIVE_REFRESH_INTERVAL"))
        return (
            MarkdownPrinter(self.code_theme, refresh_interval, vertical_overflow)
            if self.markdown
            else TextPrinter(self.color)
        )

    def make_messages(self, prompt: str) -> List[Dict[str, str]]:
        raise NotImplementedError

    def handle_function_call(
        self,
        messages: List[dict[str, Any]],
        tool_call_id: str,
        name: str,
        arguments: str,
    ) -> Generator[str, None, None]:
        # Add assistant message with tool call
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ],
            }
        )

        if messages and messages[-1]["role"] == "assistant":
            yield "\n"

        dict_args = json.loads(arguments)
        joined_args = ", ".join(f'{k}="{v}"' for k, v in dict_args.items())
        yield f"> @FunctionCall `{name}({joined_args})` \n\n"

        # Be defensive: models sometimes hallucinate extra kwargs (e.g. a key named ":").
        # Passing unknown kwargs would crash the whole run.
        fn = get_function(name)
        sig = inspect.signature(fn)
        # Take only params which are in the func signature:
        allowed_param_names = set()
        for param_name in sig.parameters:
            allowed_param_names.add(param_name)
        safe_args = {k: v for k, v in dict_args.items() if k in allowed_param_names}
        result = fn(**safe_args)
        if cfg.get("SHOW_FUNCTIONS_OUTPUT") == "true":
            yield f"```text\n{result}\n```\n"

        # messages.append({"role": "function", "content": result, "name": name})
        # messages.append({"role": "developer", "content": result, "name": name})
        # Danijel: replaced first line with second to make O3 work with function calls. TODO fix properly.
        # https://github.com/TheR1D/shell_gpt/issues/717
        # But syncing now gives tool:
        # Add tool response message
        messages.append({"role": "tool", "content": result, "tool_call_id": tool_call_id})

    @cache
    def get_completion(
        self,
        model: str,
        temperature: float,
        top_p: float,
        messages: List[Dict[str, Any]],
        functions: Optional[List[Dict[str, str]]],
        caching: bool = True,
        tool_rounds_left: int = 20,
    ) -> Generator[str, None, None]:
        # Track tool calls by index first.
        # OpenAI streaming often sends tool call argument chunks with only `index`
        # (id/name missing), and later provides id/name. Index is the stable key.
        tool_calls_by_index: dict[int, dict[str, str]] = {}
        is_shell_role = self.role.name == DefaultRoles.SHELL.value
        is_code_role = self.role.name == DefaultRoles.CODE.value
        is_dsc_shell_role = self.role.name == DefaultRoles.DESCRIBE_SHELL.value
        if is_shell_role or is_code_role or is_dsc_shell_role:
            functions = None

        if functions:
            additional_kwargs["tool_choice"] = "auto"
            additional_kwargs["tools"] = functions
            # Danijel: not supported by new o3 model, so comment out
            # additional_kwargs["parallel_tool_calls"] = False

        response = completion(
            model=model,
            temperature=temperature,
            top_p=top_p,
            messages=messages,
            stream=True,
            **additional_kwargs,
        )

        try:
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # Debug output: keep it minimal (only print at execution time).

                # LiteLLM uses dict instead of Pydantic object like OpenAI does.
                tool_calls = (
                    delta.get("tool_calls") if use_litellm else delta.tool_calls
                )
                if tool_calls:
                    for tool_call in tool_calls:
                        if use_litellm:
                            # LiteLLM returns dict-like deltas.
                            tc_id = tool_call.get("id")
                            tc_index = tool_call.get("index")
                            fn = tool_call.get("function", {}) or {}
                            fn_name = fn.get("name")
                            fn_args_chunk = fn.get("arguments", "") or ""
                        else:
                            # OpenAI streams tool calls in small pieces.
                            # In early chunks you may only get:
                            #   - tool_call.index
                            #   - function.arguments (partial JSON/text)
                            # and id/name often arrive later.
                            # That's why we key on index first.
                            tc_id = tool_call.id
                            tc_index = getattr(tool_call, "index", None)
                            fn_name = None
                            fn_args_chunk = ""
                            if getattr(tool_call, "function", None) is not None:
                                fn_name = getattr(tool_call.function, "name", None)
                                fn_args_chunk = getattr(tool_call.function, "arguments", None) or ""

                        if tc_index is None:
                            # No index -> we cannot correlate chunks -> ignore.
                            continue

                        # Accumulate state for this tool call index.
                        # State fields are filled as they become available.
                        state = tool_calls_by_index.setdefault(int(tc_index), {"id": "", "name": "", "arguments": ""})
                        if tc_id:
                            # Adopt the final tool_call_id once provided by the model.
                            state["id"] = tc_id
                        if fn_name:
                            # Adopt the tool/function name once provided.
                            state["name"] = fn_name
                        if fn_args_chunk:
                            # Append partial argument chunks.
                            # The model streams JSON args as plain text pieces.
                            state["arguments"] += fn_args_chunk

                if chunk.choices[0].finish_reason == "tool_calls":
                    if tool_rounds_left <= 0:
                        yield "\nERROR: reached max tool rounds (20). Stopping.\n\n"
                        return

                    # Ensure we start clean for execution. If a previous round left any
                    # partial state around, nuke it.
                    # (Shouldn't happen, but makes the function more robust.)
                    # NOTE: we clear again after executing tool calls.

                    # The assistant has finished emitting tool calls.
                    # Now we must execute tools and append a `role="tool"` message for each one.
                    # OpenAI API requirement:
                    #   - An assistant message with `tool_calls` MUST be followed by tool messages
                    #     responding to EACH `tool_call_id`.
                    # Strategy:
                    #   - Execute in stable index order (0,1,2,...)
                    #   - Always respond using the *real* tool_call_id when present.
                    for idx, state in sorted(tool_calls_by_index.items()):
                        tc_id = state.get("id")
                        name = state.get("name")
                        arguments = state.get("arguments", "")
                        if not tc_id or not name:
                            # If the model emitted an incomplete tool call, don't recurse forever.
                            # Stop here so the user sees something is wrong.
                            yield "\nERROR: incomplete tool call (missing id or name).\n\n"
                            tool_calls_by_index.clear()
                            return
                        if not arguments.strip():
                            yield f"\nERROR: tool call '{name}' has empty arguments.\n\n"
                            tool_calls_by_index.clear()
                            return
                        args_preview = arguments.replace("\n", "\\n")
                        if len(args_preview) > 400:
                            args_preview = args_preview[:400] + "..."
                        yield from self.handle_function_call(
                            messages,
                            tc_id,
                            name,
                            arguments,
                        )

                    yield "\n"
                    for _ in range(3):
                        yield "---\n"  # markdown horizontal rule; renders nicely

                    tool_calls_by_index.clear()

                    # Re-enter the model with tool outputs appended to `messages`.
                    # This allows the assistant to continue from where it left off.
                    yield from self.get_completion(
                        model=model,
                        temperature=temperature,
                        top_p=top_p,
                        messages=messages,
                        functions=functions,
                        caching=caching,
                        tool_rounds_left=tool_rounds_left - 1,
                    )
                    return

                yield delta.content or ""
        except KeyboardInterrupt:
            response.close()

    def handle(
        self,
        prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        caching: bool,
        functions: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> str:
        disable_stream = cfg.get("DISABLE_STREAMING") == "true"
        messages = self.make_messages(prompt.strip())
        generator = self.get_completion(
            model=model,
            temperature=temperature,
            top_p=top_p,
            messages=messages,
            functions=functions,
            caching=caching,
            **kwargs,
        )
        return self.printer(generator, not disable_stream)
