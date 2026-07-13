"""In-process SGLang compatibility patches (no edits to installed packages)."""

from __future__ import annotations


def _patch_sglang_autoconfig_exist_ok() -> list[str]:
    """Monkey-patch sglang AutoConfig.register to accept exist_ok clashes."""
    try:
        from transformers import AutoConfig
    except Exception:
        return []

    original = getattr(AutoConfig, "register", None)
    if original is None or getattr(original, "_simple_wiki_exist_ok_patched", False):
        return []

    def register(model_type, config, exist_ok: bool = False):  # type: ignore[no-untyped-def]
        try:
            return original(model_type, config, exist_ok=exist_ok)
        except ValueError as exc:
            if "already used by a Transformers config" not in str(exc):
                raise
            return original(model_type, config, exist_ok=True)

    register._simple_wiki_exist_ok_patched = True  # type: ignore[attr-defined]
    AutoConfig.register = register  # type: ignore[method-assign]
    return ["transformers.AutoConfig.register"]


def _patch_pixtral_skip_redundant_pad_token() -> list[str]:
    """Skip PixtralProcessor pad_token add when already set / MistralCommonBackend.

    transformers>=5 MistralCommonBackend raises NotImplementedError on
    add_special_tokens; Ministral already has pad_token=<pad>, so the call is
    redundant.
    """
    try:
        from sglang.srt.multimodal.processors.pixtral import PixtralProcessor
        from transformers import PreTrainedTokenizerBase
    except Exception:
        return []

    if getattr(PixtralProcessor, "_simple_wiki_skip_pad_patched", False):
        return []

    original_init = PixtralProcessor.__init__

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # type: ignore[no-untyped-def]
        tokenizer = (
            _processor
            if isinstance(_processor, PreTrainedTokenizerBase)
            else getattr(_processor, "tokenizer", None)
        )
        restore = None
        if tokenizer is not None:
            desired_pad = getattr(hf_config, "pad_token", PixtralProcessor.PAD_TOKEN)
            already_ok = getattr(tokenizer, "pad_token", None) == desired_pad
            is_mistral_common = "MistralCommon" in type(tokenizer).__name__
            if already_ok or is_mistral_common:
                original_add = tokenizer.add_special_tokens

                def _noop_add_special_tokens(*_a, **_k):  # type: ignore[no-untyped-def]
                    return 0

                tokenizer.add_special_tokens = _noop_add_special_tokens  # type: ignore[method-assign]
                restore = (tokenizer, original_add)
        try:
            return original_init(
                self, hf_config, server_args, _processor, *args, **kwargs
            )
        finally:
            if restore is not None:
                tok, original_add = restore
                tok.add_special_tokens = original_add  # type: ignore[method-assign]

    PixtralProcessor.__init__ = __init__  # type: ignore[method-assign]
    PixtralProcessor._simple_wiki_skip_pad_patched = True  # type: ignore[attr-defined]
    return ["sglang.PixtralProcessor.skip_redundant_pad_token"]


def _patch_sglang_tool_omit_defer_loading() -> list[str]:
    """Omit ``defer_loading=None`` from SGLang Tool dumps for mistral_common.

    sglang Tool.model_dump() always emits ``defer_loading`` (often None).
    MistralCommonBackend -> ChatCompletionRequest.from_openai rejects unknown
    fields, then serving_chat retries with flat function-only tools which also
    fail Tool validation (needs ``function`` wrapper). Symptom: 400
    ``validation errors for Tool`` / ``Field required: function``.
    Function already strips None defer_loading via model_serializer; Tool does not.
    """
    try:
        from sglang.srt.entrypoints.openai import protocol as proto
    except Exception:
        return []

    if getattr(proto.Tool, "_simple_wiki_defer_loading_patched", False):
        return []

    original_dump = proto.Tool.model_dump

    def model_dump(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        data = original_dump(self, *args, **kwargs)
        if isinstance(data, dict) and data.get("defer_loading") is None:
            data.pop("defer_loading", None)
        return data

    proto.Tool.model_dump = model_dump  # type: ignore[method-assign]
    proto.Tool._simple_wiki_defer_loading_patched = True  # type: ignore[attr-defined]
    return ["sglang.openai.Tool.omit_defer_loading"]


def _patch_mistral_common_preserve_special_token_ids() -> list[str]:
    """Keep MistralCommon special-token ids when sglang uses tokenize=False+encode.

    sglang serving_chat renders with ``apply_chat_template(tokenize=False)`` then
    ``encode()``. For MistralCommonBackend that round-trip turns control tokens
    like ``[/INST]`` / ``[AVAILABLE_TOOLS]`` into plain text pieces, so tool
    prompts EOS immediately (empty completion).

    Patch the class methods directly: sglang imports
    ``patch_mistral_common_tokenizer`` by name at module load, so wrapping the
    mistral_utils symbol alone does not reach already-bound call sites.
    """
    try:
        from transformers.tokenization_mistral_common import MistralCommonBackend
    except Exception:
        return []

    if getattr(MistralCommonBackend, "_simple_wiki_special_ids_patched", False):
        return []

    original_apply = MistralCommonBackend.apply_chat_template
    original_encode = MistralCommonBackend.encode

    def apply_chat_template(self, conversation, *args, **kwargs):  # type: ignore[no-untyped-def]
        tokenize = kwargs.get("tokenize", True)
        if tokenize is False:
            id_kwargs = dict(kwargs)
            id_kwargs["tokenize"] = True
            id_kwargs["return_dict"] = False
            prompt_ids = original_apply(self, conversation, *args, **id_kwargs)
            if hasattr(prompt_ids, "input_ids"):
                prompt_ids = prompt_ids["input_ids"]
            if not isinstance(prompt_ids, list):
                prompt_ids = list(prompt_ids)
            text = original_apply(self, conversation, *args, **kwargs)
            self._simple_wiki_last_prompt_text = text
            self._simple_wiki_last_prompt_ids = prompt_ids
            return text
        return original_apply(self, conversation, *args, **kwargs)

    def encode(self, text, *args, **kwargs):  # type: ignore[no-untyped-def]
        cached_text = getattr(self, "_simple_wiki_last_prompt_text", None)
        cached_ids = getattr(self, "_simple_wiki_last_prompt_ids", None)
        if cached_ids is not None and text == cached_text:
            return list(cached_ids)
        return original_encode(self, text, *args, **kwargs)

    MistralCommonBackend.apply_chat_template = apply_chat_template  # type: ignore[method-assign]
    MistralCommonBackend.encode = encode  # type: ignore[method-assign]
    MistralCommonBackend._simple_wiki_special_ids_patched = True  # type: ignore[attr-defined]
    return ["transformers.MistralCommonBackend.preserve_special_token_ids"]


def _patch_ministral3_start_layer_kwargs() -> list[str]:
    """Fix Ministral3 positional-arg mismatch after Llama gained start_layer.

    sglang>=0.5.13 inserted ``start_layer`` into LlamaAttention/LlamaDecoderLayer.
    Ministral3 still calls ``super().__init__(..., layer_id, quant_config, prefix)``,
    so ``prefix`` (str) is mis-bound as ``quant_config`` and crashes with
    ``'str' object has no attribute 'get_quant_method'`` (sglang#29835).
    """
    try:
        from sglang.srt.models import ministral3 as m3
        from sglang.srt.utils import add_prefix
    except Exception:
        return []

    if getattr(m3.Ministral3DecoderLayer, "_simple_wiki_start_layer_patched", False):
        return []

    def attention_init(  # type: ignore[no-untyped-def]
        self,
        config,
        hidden_size,
        num_heads,
        num_kv_heads,
        layer_id=0,
        rope_theta=1000000.0,
        rope_scaling=None,
        rope_is_neox_style=True,
        max_position_embeddings=8192,
        quant_config=None,
        prefix="",
        bias=False,
    ):
        if rope_scaling is None:
            rope_scaling = {}
        m3.LlamaAttention.__init__(
            self,
            config,
            hidden_size,
            num_heads,
            num_kv_heads,
            layer_id,
            start_layer=0,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_is_neox_style=rope_is_neox_style,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=prefix,
            bias=bias,
        )
        self.llama_4_scaling_beta = config.rope_parameters.get("llama_4_scaling_beta")
        self.sliding_window = getattr(config, "sliding_window", None)

    def decoder_init(  # type: ignore[no-untyped-def]
        self, config, layer_id=0, quant_config=None, prefix=""
    ):
        m3.LlamaDecoderLayer.__init__(
            self,
            config,
            layer_id,
            start_layer=0,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.self_attn = m3.Ministral3Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=config.rope_parameters["rope_theta"],
            rope_scaling=config.rope_parameters,
            max_position_embeddings=getattr(
                config, "original_max_position_embeddings", 16384
            ),
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            bias=getattr(config, "attention_bias", False)
            or getattr(config, "bias", False),
        )

    m3.Ministral3Attention.__init__ = attention_init  # type: ignore[method-assign]
    m3.Ministral3DecoderLayer.__init__ = decoder_init  # type: ignore[method-assign]
    m3.Ministral3DecoderLayer._simple_wiki_start_layer_patched = True  # type: ignore[attr-defined]
    return ["sglang.Ministral3.start_layer_kwargs"]


def _patch_sglang_mistral_tool_call_ids() -> list[str]:
    """Emit mistral_common-compliant 9-char tool_call ids for --tool-call-parser mistral.

    Default SGLang ids look like ``call_<24 hex>``; mistral_common serving
    validation requires ``^[a-zA-Z0-9]{9}$``, so the *next* turn fails when those
    ids are echoed back in messages (often misreported as Tool schema errors
    after the flat-tools fallback).
    """
    try:
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    except Exception:
        return []

    if getattr(OpenAIServingChat, "_simple_wiki_mistral_tool_id_patched", False):
        return []

    original = OpenAIServingChat._process_tool_call_id

    def _process_tool_call_id(self, call_item, history_tool_calls_cnt):  # type: ignore[no-untyped-def]
        tool_call_id = original(self, call_item, history_tool_calls_cnt)
        if getattr(self, "tool_call_parser", None) != "mistral":
            return tool_call_id
        import hashlib
        import uuid

        raw = str(tool_call_id or "")
        if len(raw) == 9 and raw.isalnum():
            return raw
        seed = raw or f"{uuid.uuid4().hex}:{history_tool_calls_cnt}"
        return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:9]

    OpenAIServingChat._process_tool_call_id = _process_tool_call_id  # type: ignore[method-assign]
    OpenAIServingChat._simple_wiki_mistral_tool_id_patched = True  # type: ignore[attr-defined]
    return ["sglang.openai.mistral_tool_call_ids"]


def _patch_sglang_mistral_no_flat_tools_fallback() -> list[str]:
    """For mistral, do not retry apply_chat_template with flat function-only tools.

    Upstream catches *any* first-render failure and retries with flat tools.
    mistral_common rejects that flat shape, so clients only see a misleading
    ``validation errors for Tool`` 400 while the real first error (empty tool
    name, bad tool_call id, etc.) is discarded. Re-raise the original error.
    """
    try:
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    except Exception:
        return []

    if getattr(OpenAIServingChat, "_simple_wiki_mistral_no_flat_tools_patched", False):
        return []

    original = OpenAIServingChat._apply_jinja_template

    def _apply_jinja_template(self, request, tools, is_multimodal):  # type: ignore[no-untyped-def]
        if getattr(self, "tool_call_parser", None) != "mistral":
            return original(self, request, tools, is_multimodal)

        tokenizer = self.tokenizer_manager.tokenizer
        real_apply = tokenizer.apply_chat_template
        first_error: dict[str, BaseException] = {}

        def apply_once(*args, **kwargs):  # type: ignore[no-untyped-def]
            try:
                return real_apply(*args, **kwargs)
            except Exception as exc:
                if "err" not in first_error:
                    first_error["err"] = exc
                # On the upstream flat-tools retry, keep surfacing the first cause.
                raise first_error["err"]

        tokenizer.apply_chat_template = apply_once  # type: ignore[method-assign]
        try:
            return original(self, request, tools, is_multimodal)
        except Exception as exc:
            root = first_error.get("err", exc)
            raise ValueError(str(root)) from root
        finally:
            tokenizer.apply_chat_template = real_apply  # type: ignore[method-assign]

    OpenAIServingChat._apply_jinja_template = _apply_jinja_template  # type: ignore[method-assign]
    OpenAIServingChat._simple_wiki_mistral_no_flat_tools_patched = True  # type: ignore[attr-defined]
    return ["sglang.openai.mistral_no_flat_tools_fallback"]


def apply_sglang_compat_patches() -> list[str]:
    """Apply in-process SGLang/transformers shims (no edits to site-packages).

    Verified against: sglang==0.5.14. Re-check when upgrading; some patches may
    become obsolete or need retargeting.

    Why this exists
    ---------------
    Upstream SGLang + transformers + mistral_common combinations we run still
    have a few hard failures or silent tool-calling breakages. We monkey-patch
    at process start (``serving.sglang_launch_server`` / ``llm_router``
    preflight) instead of forking those packages.

    What each patch fixes / who it hits
    -----------------------------------
    - ``AutoConfig.register`` exist_ok:
      Config type already registered → import crash. Mostly Gemma-4 / newer
      transformers×sglang clashes; harmless no-op when unused.
    - ``MistralCommonBackend`` special-token preserve:
      ``apply_chat_template(tokenize=False)`` + ``encode()`` drops control
      tokens → empty tool completions. Affects Mistral/Ministral Instruct
      that use MistralCommonBackend (incl. vision Ministral via Pixtral path).
    - ``PixtralProcessor`` skip redundant pad_token:
      transformers≥5 MistralCommonBackend cannot ``add_special_tokens``;
      Ministral already has ``<pad>``. Affects Pixtral/Ministral multimodal
      load only.
    - ``Tool.omit_defer_loading``:
      SGLang dumps ``defer_loading=None``; mistral_common rejects it, then
      flat-tools fallback also fails → 400 ``validation errors for Tool``.
      Affects any chat that goes through mistral_common tool schemas
      (``--tool-call-parser mistral``).
    - ``mistral_tool_call_ids``:
      Default SGLang ids are ``call_<24hex>``; mistral_common requires
      ``^[a-zA-Z0-9]{9}$``, so multi-turn tool history 400s (often mislabeled
      as Tool schema errors). Only remaps when ``tool_call_parser == "mistral"``
      (Ministral/Mistral Instruct). Qwen/Gemma parsers unchanged.
    - ``mistral_no_flat_tools_fallback``:
      Upstream flat-tools retry masks the real first render error behind a
      Tool-schema 400. For ``tool_call_parser=mistral``, re-raise the original
      cause (e.g. illegal function name / tool_call id).
    - ``Ministral3.start_layer_kwargs``:
      sglang≥0.5.13 Llama layers gained ``start_layer``; Ministral3 still
      passes positional args and crashes on load. Ministral-3 weights only.

    Non-goals: do not rewrite bad model tool outputs into plausible ones.
    Surface the real validator error; the agent may turn that into an
    observation for the next round.
    """
    patched: list[str] = []
    patched.extend(_patch_sglang_autoconfig_exist_ok())
    patched.extend(_patch_mistral_common_preserve_special_token_ids())
    patched.extend(_patch_pixtral_skip_redundant_pad_token())
    patched.extend(_patch_sglang_tool_omit_defer_loading())
    patched.extend(_patch_sglang_mistral_tool_call_ids())
    patched.extend(_patch_sglang_mistral_no_flat_tools_fallback())
    patched.extend(_patch_ministral3_start_layer_kwargs())
    return patched
