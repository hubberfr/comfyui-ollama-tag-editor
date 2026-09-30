"""
OllamaTagEditor - 极简 ComfyUI 节点
输入：WD Tagger 给出的原图标签 + 用户修改要求(中英文均可)
输出：模型按要求编辑后的标签串（单 STRING，可直接接 CLIPTextEncode）
无第三方依赖（urllib 直连 Ollama /api/chat）
模型下拉框：自动从 {base_url}/api/tags 拉取本机已装模型（60 秒缓存，服务不可用时用内置候选）
稳定性：强制关闭思考(think:false) + 重复惩罚 + 复读去重 + 超时保护 + 复读自动重试一次
调试：每次运行都会把 指令/原标签/模型原始返回/最终标签 打印到 ComfyUI 后端控制台

注意：新增控件一律追加到参数列表【末尾】，避免打乱老工作流按位置保存的控件值。
"""
import json
import re
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://127.0.0.1:11434"

# 拉取失败时的兜底候选，保证下拉框永远有可选项、老工作流不会失效
FALLBACK_MODELS = [
    "nanbeige42:3b",
    "hf.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive:Q4_K_M",
]

_MODEL_CACHE = {"ts": 0.0, "models": []}
_MODEL_CACHE_TTL = 60.0  # 秒：新增模型后，刷新/重开节点定义即可看到

SYSTEM_PROMPT = (
    "You are an expert editor of Danbooru tags for an anime diffusion model. "
    "You receive (1) original_tags: tags auto-detected from an input image, and (2) an Instruction "
    "describing a change the user wants.\n"
    "Your task: output an EDITED tag list that:\n"
    "- keeps every original tag that does NOT conflict with the change;\n"
    "- REMOVES original tags that conflict with the change (e.g. old dress/outfit tags when the user "
    "asks for a new outfit);\n"
    "- applies the requested change using real, lowercase Danbooru tags separated by spaces inside a tag "
    "('black dress' not 'black_dress');\n"
    "- when the Instruction changes an outfit or its color, ALSO replace the color of every garment that "
    "belongs to that outfit (e.g. change 'white dress'->'black dress' AND 'white skirt'->'black skirt', "
    "'white shirt'->'black shirt', 'white footwear'->'black footwear' when present), so no old-color "
    "clothing tags contradict the new outfit; leave truly separate accessories (e.g. headwear, glasses) "
    "unchanged unless the Instruction mentions them;\n"
    "- never adds quality tags like masterpiece, best quality, absurdres, score_9;\n"
    "- place the tags representing the requested CHANGE (e.g. 'black dress', 'black skirt') immediately "
    "after the leading people/character tags and BEFORE unrelated general tags, so the change is "
    "emphasized in the prompt;\n"
    "- NEVER repeat a tag; each tag must appear at most once;\n"
    "- output ONLY the comma-separated tag list, no explanations, no numbering, no surrounding text.\n"
    "Example: original_tags contain 'white dress', 'white skirt', 'white shirt' and Instruction says 换黑色连衣裙 "
    "-> remove them and add 'black dress', 'black skirt', 'black shirt'.\n"
    "The Instruction may be written in Chinese; understand the meaning but always output tags in English."
)

QUALITY_TAGS = ("masterpiece", "best quality", "amazing quality", "very aesthetic",
                "absurdres", "high resolution", "ultra-detailed", "score_9",
                "score_8_up", "score_7_up", "newest", "highres")


def list_ollama_models(base_url=DEFAULT_BASE_URL, ttl=_MODEL_CACHE_TTL):
    """列出 Ollama 已安装模型；结果带缓存，失败时回退到内置候选。"""
    now = time.time()
    if now - _MODEL_CACHE["ts"] < ttl and _MODEL_CACHE["models"]:
        return _MODEL_CACHE["models"]

    base = (base_url or DEFAULT_BASE_URL).rstrip("/")
    names = []
    try:
        with urllib.request.urlopen(base + "/api/tags", timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        names = sorted(m["name"] for m in data.get("models", []) if m.get("name"))
    except Exception as e:
        print(f"[OllamaTagEditor] 读取 {base}/api/tags 失败({e})，使用内置候选模型列表")

    for m in FALLBACK_MODELS:
        if m not in names:
            names.append(m)

    _MODEL_CACHE["ts"] = now
    _MODEL_CACHE["models"] = names
    return names


def _as_float(value, default, name):
    try:
        return float(value)
    except (TypeError, ValueError):
        print(f"[OllamaTagEditor] 参数 {name}={value!r} 不是数值，已改用默认 {default}")
        return float(default)


def _as_int(value, default, name):
    try:
        return int(value)
    except (TypeError, ValueError):
        print(f"[OllamaTagEditor] 参数 {name}={value!r} 不是整数，已改用默认 {default}")
        return int(default)


class OllamaTagEditor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_tags": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "WD Tagger 输出的原图标签(逗号分隔)，可留空",
                }),
                "instruction": ("STRING", {
                    "multiline": True,
                    "default": "换一身黑色连衣裙，其他保持不变",
                    "tooltip": "你想怎么改(中英文均可)",
                }),
                "model": (list_ollama_models(), {
                    "tooltip": "自动从 Ollama(/api/tags) 拉取的已装模型。新装模型后：右键节点→刷新节点定义，或重开 ComfyUI。",
                }),
                "base_url": ("STRING", {
                    "default": DEFAULT_BASE_URL,
                }),
                "temperature": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "max_tokens": ("INT", {
                    "default": 1024, "min": 64, "max": 8192, "step": 64,
                    "tooltip": "生成长度上限。标签任务 1024 足够；调小可让复读更早被截断。",
                }),
                "quality_suffix": ("STRING", {
                    "default": "masterpiece, best quality, absurdres",
                    "multiline": False,
                    "tooltip": "自动追加到标签末尾的质量词。Illustrious/NoobAI 系必须有；留空则不追加。",
                }),
                "repeat_penalty": ("FLOAT", {
                    "default": 1.15, "min": 1.0, "max": 2.0, "step": 0.05,
                    "tooltip": "重复惩罚。小模型偶发'复读循环'时调大到 1.15~1.3 可显著抑制。",
                }),
                # ↓↓↓ 新增控件一律追加在末尾，避免打乱老工作流按位置保存的控件值 ↓↓↓
                "timeout": ("INT", {
                    "default": 90, "min": 10, "max": 600, "step": 10,
                    "tooltip": "单次请求超时(秒)。超时直接报错，避免模型复读时看起来'卡死'。",
                }),
                "retry_on_loop": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "检测到复读/截断时，自动用更高重复惩罚重试一次。",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    FUNCTION = "edit"
    CATEGORY = "tag"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # NaN => 每次 Queue 都重新调用 Ollama，避免被 ComfyUI 缓存旧结果
        return float("NaN")

    def _call(self, base, model, original_tags, instruction, temperature, max_tokens,
              repeat_penalty, timeout):
        payload = {
            "model": model,
            "stream": False,
            "think": False,  # Qwen3 / Nanbeige4.2 等模板默认思考，必须关闭，否则 content 可能为空
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "repeat_penalty": repeat_penalty,
                "repeat_last_n": 256,
                "stop": ["\n\n"],
            },
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"Original tags: {original_tags}\n\nInstruction: {instruction}"},
            ],
        }
        req = urllib.request.Request(
            base + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
        except Exception as e:
            raise RuntimeError(f"Ollama 请求失败({base}, 超时={timeout}s): {e}")
        content = (data.get("message") or {}).get("content", "") or ""
        return content, data.get("done_reason")

    def _parse(self, content):
        """原始返回 -> (全部标签, 去重后标签)"""
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
        content = content.replace("/no_think", "")
        tokens = [t.strip() for t in re.split(r"[\n,]+", content) if t.strip()]
        tokens = [t for t in tokens
                  if not t.startswith("```") and t.lower() not in ("original tags:", "instruction:")]
        tokens = [t for t in tokens if t.lower() not in QUALITY_TAGS]
        seen, deduped = set(), []
        for t in tokens:
            key = t.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(t)
        return tokens, deduped

    def edit(self, original_tags, instruction, model, base_url, temperature, max_tokens,
             quality_suffix="masterpiece, best quality, absurdres", repeat_penalty=1.15,
             timeout=90, retry_on_loop=True):
        temperature = _as_float(temperature, 0.3, "temperature")
        max_tokens = _as_int(max_tokens, 1024, "max_tokens")
        repeat_penalty = _as_float(repeat_penalty, 1.15, "repeat_penalty")
        timeout = _as_int(timeout, 90, "timeout")
        original_tags = "" if original_tags is None else str(original_tags)
        instruction = "" if instruction is None else str(instruction)
        base_url = str(base_url or DEFAULT_BASE_URL).rstrip("/")
        model = str(model)

        print(f"[OllamaTagEditor] === 开始 (model={model}) ===")
        print(f"[OllamaTagEditor] instruction: {instruction}")
        print(f"[OllamaTagEditor] original_tags(前400字): {original_tags[:400]}")

        content, done_reason = self._call(base_url, model, original_tags, instruction,
                                         temperature, max_tokens, repeat_penalty, timeout)
        print(f"[OllamaTagEditor] 模型原始返回(前600字): {content[:600]}")
        if done_reason == "length":
            print("[OllamaTagEditor] 警告: 达到 max_tokens 上限被截断")

        tokens, deduped = self._parse(content)
        degenerate = (not deduped) or (len(tokens) > 0 and len(deduped) < 0.6 * len(tokens)) \
            or done_reason == "length"

        if degenerate and retry_on_loop:
            new_penalty = min(1.6, repeat_penalty + 0.15)
            new_temp = max(0.05, temperature - 0.1)
            print(f"[OllamaTagEditor] 检测到异常输出(标签{len(tokens)}个/去重后{len(deduped)}个, done={done_reason})，"
                  f"用 repeat_penalty={new_penalty:.2f} 重试一次")
            try:
                content2, done2 = self._call(base_url, model, original_tags, instruction,
                                            new_temp, max_tokens, new_penalty, timeout)
                tokens2, deduped2 = self._parse(content2)
                if len(deduped2) > len(deduped):
                    print(f"[OllamaTagEditor] 重试成功: 标签 {len(deduped)} -> {len(deduped2)} 个")
                    tokens, deduped, done_reason = tokens2, deduped2, done2
            except Exception as e:
                print(f"[OllamaTagEditor] 重试失败，沿用首次结果: {e}")

        if not deduped:
            raise RuntimeError(f"模型未产出有效标签(model={model})，原始返回: {content[:200]!r}")
        if len(tokens) != len(deduped):
            print(f"[OllamaTagEditor] 复读去重: {len(tokens)} -> {len(deduped)} 个标签")
        if len(deduped) > 200:
            print(f"[OllamaTagEditor] 标签过多({len(deduped)})，截取前 200 个")
            deduped = deduped[:200]

        result = ", ".join(deduped)
        suffix = str(quality_suffix).strip().strip(",") if quality_suffix else ""
        if suffix:
            result = f"{result}, {suffix}" if result else suffix
        print(f"[OllamaTagEditor] 质量词后缀: {suffix if suffix else '(无)'}")
        print(f"[OllamaTagEditor] 最终标签: {result}")
        return (result,)


NODE_CLASS_MAPPINGS = {"OllamaTagEditor": OllamaTagEditor}
NODE_DISPLAY_NAME_MAPPINGS = {"OllamaTagEditor": "Ollama Tag Editor (改标签)"}
