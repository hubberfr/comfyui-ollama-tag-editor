# 排障记录：四个真实故障的定位与修复

> 环境：Windows · ComfyUI（aki 整合包）· RTX 4060 Laptop 8GB · Ollama · Python 3.13
> 记录方式统一为：**现象 → 定位手段 → 根因 → 修复 → 验证**。这份文档也是"如何在没有报错的情况下定位问题"的示例。

---

## 案例 1：画面颜色异常鲜艳、画风诡异，且调 CFG 完全无效

**现象**
新下载的 `NoobAI-XL-Vpred-v1.0.safetensors` 出图颜色过饱和、画风崩坏；把 CFG 从 5 调到 4.5 / 7 都不改善。

**定位手段**
读取 safetensors 文件头部的 `__metadata__`（safetensors 的前 8 字节是 header 长度，其后是 JSON）：

```python
import json, struct

path = r"...\models\checkpoints\NoobAI-XL-Vpred-v1.0.safetensors"
with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]      # header 长度
    header = json.loads(f.read(n).decode("utf-8"))

print(header.get("__metadata__"))
# 输出：{'current_epoch': 2, 'global_step': 10855}
```

正常情况下 v-pred 检查点应带有 `modelspec.prediction_type = v_prediction` 之类的标记。
ComfyUI 依赖该标记决定采样时的预测类型；**没有标记 → 按 eps 采样 v-pred 权重**。

**根因**
预测类型（eps / v-pred）决定了模型输出被如何解释。用 eps 的语义去解读 v-pred 的输出，
去噪方向错误 → 过饱和、结构崩坏。而 CFG 只控制引导强度，**改不了这个根本性错误**，所以"调 CFG 没用"。

**修复**
在 Checkpoint 与 KSampler 之间插入采样类型节点：

```
CheckpointLoaderSimple ──MODEL──▶ ModelSamplingDiscrete(sampling=v_prediction) ──▶ KSampler.model
```

- `sampling = v_prediction`
- `zsnr` 需实测：SDXL v-pred 惯例为 `true`；若出现发灰/发糊则改 `false` 再比对
- 另外 v-pred 只能用 **Euler / DDIM**，不能用 Karras 系调度器

**验证**
修复后画面恢复正常；同时可在控制台观察模型加载行打印的模型类型（此前与修复后不同）。

---

## 案例 2：换了模型后输出"发冲"、质量明显下降（同一套流水线）

**现象**
同一张参考图、同一句改写指令，NoobAI 分支输出异常，而手写提示词的对照工作流却正常。

**定位手段**
两个工作流的唯一差异是**正向提示词的来源**：

| | 手写提示词工作流 | 流水线工作流 |
|---|---|---|
| 正向内容 | 含 `masterpiece, best quality, absurdres` | 打标 + LLM 改写，**且节点系统提示词明确禁止 LLM 输出质量词** |

用"同 seed + 同底噪 + 只换模型"的 A/B 工作流隔离变量后，问题稳定复现在"缺质量词"的分支。

**根因**
Illustrious / NoobAI 这类在 Danbooru 数据上训练的模型**高度依赖质量词**；
缺失时会表现为对比失控、画面发冲——属于"提示词侧"问题，不是模型坏。

**修复**
在标签编辑节点增加 `quality_suffix`（默认 `masterpiece, best quality, absurdres`）：
输出标签后自动追加，并剔除 LLM 万一自带的质量词避免重复。

**验证**
同一指令下，两个模型的分支均恢复正常；A/B 对比工作流的两张图可用于人工比对。

---

## 案例 3：LLM 输出"复读"——同一标签重复 200+ 次直至截断

**现象**
控制台打印的原始返回尾部出现几十上百个 `bare legs`，最终标签被截断成半截词：

```
..., bare legs, bare legs, bare legs, ..., bare, masterpiece, best quality, absurdres
```

**定位手段**
节点把 `instruction / original_tags / 模型原始返回 / done_reason` 全部打到控制台。
观察到 `done_reason = length` → 说明模型耗尽 `num_predict` 预算仍未停止，
再结合返回文本即可确认是**采样退化复读**，而不是提示词问题。

**根因**
小参数量的社区微调（尤其去除审查类）在低熵场景下容易进入重复循环；
一旦循环，生成预算被"同一个词"耗尽，正文被截断。
仅靠"把原始返回按逗号拼接"的朴素实现，会把上百个重复标签全部塞进提示词。

**修复（五道防护，缺一不可）**

| 防护 | 实现 |
|---|---|
| 重复惩罚 | `options.repeat_penalty = 1.15`（可调 1.15~1.3）、`repeat_last_n = 256` |
| 提前中断 | `options.stop = ["\n\n"]` |
| 输出去重 | 按首次出现顺序去重（保留原来的标签顺序） |
| 长度上限 | 超过 200 个标签截断，避免病态长提示词挤占 CLIP 额度 |
| 超时与重试 | 请求超时（默认 90s）+ 检测到"去重率过低 / done_reason=length"时用更高惩罚自动重试一次 |

**验证**
控制台会打印 `检测到复读并已去重: N -> M 个标签`。
若要量化，可固定一条指令跑 50~100 次，统计"复读率 / 格式合规率 / 平均耗时"。

---

## 案例 4：工作流加载后参数错位，报"无法将 xxx 转换为期望类型"

**现象**
节点新增了一个参数后，之前保存的工作流一运行就报错：

```
无法将 repeat_penalty 转换为期望类型
```

**定位手段**
对照节点当前的控件顺序与工作流存档里的控件值：
旧存档按**位置**保存控件值（数组），我在中间插入新参数后，
原本属于 `quality_suffix` 的字符串 `"masterpiece, best quality, absurdres"` 被喂给了 FLOAT 类型的 `repeat_penalty`。

**根因**
ComfyUI 的控件值按顺序映射；**在参数列表中间插入新控件会让所有旧存档错位**。

**修复（两条一起做）**

1. **新增控件一律追加到参数列表末尾**（本条已写入节点源码注释，作为长期约定）；
2. **参数转换容错**：三个数值参数（temperature / max_tokens / repeat_penalty）在入口做
   `try: float(v) / int(v) except (TypeError, ValueError): 用默认值并打印警告`，
   使脏值只降级、不中断工作流。

**验证**
用故意错位的输入调用节点：`temperature="masterpiece"`、`max_tokens=None`、`repeat_penalty="not-a-number"`，
节点打印三条降级警告并继续正常出图。

---

## 附：环境类问题（非代码缺陷，但同样耗时）

| 现象 | 原因 | 处理 |
|---|---|---|
| `SSL: CERTIFICATE_VERIFY_FAILED` 下载模型失败 | 网络中间设备拦截 TLS | 改用镜像源；不要用"关闭证书校验"绕过 |
| 大模型下载中途超时、速度极慢 | 新版客户端默认走 Xet 分块传输（数据来自海外 CDN） | `HF_HUB_DISABLE_XET=1` 回退普通 HTTP；或直接用多线程下载器拉镜像直链（支持断点续传） |
| `git clone` 返回 502 / 无法访问 GitHub | 网络波动 | 重试或使用 GitHub 加速镜像前缀克隆 |
| LLM 与绘图模型互相挤显存 | 8GB 显存要同时驻留两个模型 | 量化模型 + `OLLAMA_KEEP_ALIVE=30s`（用后即卸载）+ 串行执行 |
