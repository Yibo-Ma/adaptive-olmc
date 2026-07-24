# Baselines & metric specification

> **Role:** the single source of truth for every comparison number — which baselines to run,
> how each rate/ratio is defined, and the exact commands. Consumed by the headline stage of
> the [RUNBOOK](RUNBOOK.md). (Reference notes; kept in the authors' working language.)

Source of truth for every comparison number in the paper. Aligned with the paper's
positioning: **the main result is the controlled delta (online vs static, same model /
data / size); absolute values only *locate* us in the SOTA band and are reported with the
model size shown — we never claim to beat a 70B on absolute rate.**

---

## 0. 三条原则（决定每个数字怎么算）

1. **主张 = controlled delta.** 每模态的核心结果是 `Δ% = (static_rate − online_rate)/static_rate`，在**固定 base model + 固定数据 + 固定 size** 下。model/size 不同**不进入**这个比较。
2. **定位 = 共同 benchmark.** 把 [传统 + 引用的 LLM 数字 + 你的 static(=前作) + 你的 online] 放一张表，**每行标 model size**，证明"小 40× 的模型 + online 进入 7B 区间"，不写"击败"。
3. **口径 = raw，不含模型大小.** 你的 online 传 0 bit，所以你和所有 baseline 都用 **raw rate**（不算模型参数）。绝不用 Delétang 的 adjusted rate。

## 1. 指标口径（统一）

- 三个可互换指标，全部**相对同一个 canonical raw**（见下）：
  - `rate% = 压缩后/原始 ×100`（↓，Delétang 用）
  - `ratio = 原始/压缩后`（↑，你项目 + base 论文用）= `100/rate`
  - `bpb / bpc / bpsp / bps = 压缩bits/原始单位`（↓）= `rate×0.08`（over bytes）
- **核心量 = `Δ%`**（online vs static），它自动归一化掉 model/data/size，跨一切可比。
- **canonical raw（分母）每模态固定，所有 baseline 与你的方法必须用同一个**：
  - text: 原始 **UTF-8 字节**
  - image: 原始 **RGB 像素字节 = W×H×3**
  - audio: 原始 **PCM 采样**（⚠️ 见 §5 的 bit-depth 对齐坑）
- **chunked vs unchunked 要一致**：和 LLM 比 → 传统压缩用 **chunked（2048B）** 数字；报传统"真实能力" → 用 **unchunked（整文件）**。两者分列，别混。

## 2. 数据集矩阵（两组，缺一不可）

| 用途 | text | image | audio |
|---|---|---|---|
| **G1 · Headline**（你的域，主结果 + delta 曲线） | MeDAL, Pile-of-Law/eurlex | CLIC2024, USC-textures, DIV2K | LibriSpeech, LJSpeech, **MAESTRO** |
| **G2 · Common benchmark**（对齐引用数字） | **enwik9** | **ImageNet 32×32** | **LibriSpeech**（与 G1 重叠） |

> 没有 G2 就无法和 Chinchilla/OmniZip/P2LLM 并列——它们只在 enwik9/ImageNet/LibriSpeech 上报数。

---

## 3. TEXT

| Baseline | 类型 | 来源 | 数据 | 重现? | 优先级 |
|---|---|---|---|---|---|
| gzip / zlib | 传统 · 锚点 | `gzip -9` | G1+G2 | 跑工具 | **P0** |
| brotli (Google) | 传统 · 现代 web | `brotli -q 11` | G1+G2 | 跑工具 | **P0** |
| xz / LZMA2 | 传统 · 最强**实用**通用(≈2 阶 Shannon;Chinchilla 用;**非 SOTA**) | `xz -9e` | G1+G2 | 跑工具 | **P0** |
| ~~bzip2 / zstd / cmix~~ | 冗余/小众 | — | — | — | ✂️ 删(bzip2 夹在中间;zstd/cmix 无信息或太慢) |
| **Chinchilla 1B/7B/70B** | LLM | **引用 Delétang Table 1** | G2 | ❌闭源,引用 | **P0** |
| Llama2-7B | LLM | 引用 Delétang | G2 | ❌引用 | P1 |
| FineZip / LLMZip / OmniZip | LLM | 引用各论文(enwik9) | G2 | 优先引用 | P2 |
| **LMCompress(前作) = 你的 static** | LLM | **你的 repo（`--mode static`）** | G1+G2 | ✅内置 | **P0** |
| **Ours (online)** | LLM | 你的 repo（`--mode online`） | G1+G2 | ✅你的方法 | **P0** |

## 4. IMAGE

| Baseline | 类型 | 来源 | 数据 | 重现? | 优先级 |
|---|---|---|---|---|---|
| PNG | 传统 · 锚点 | `optipng -o7` 或 ffmpeg(已有) | G1+G2 | 跑工具 | **P0** |
| WebP-lossless | 传统 · 现代(Google) | `cwebp -lossless -q 100 -m 6` | G1+G2 | 跑工具(部分已有) | **P0** |
| **JPEG-XL** | 传统 · **非神经 SOTA** | `cjxl -d 0 -e 9` | G1+G2 | 跑工具 | **P0** |
| ~~JPEG-2000 / FLIF / zstd~~ | 非SOTA或冗余 | — | — | — | ✂️ 删(JPEG-2000 弱于 WebP/JXL;FLIF 已被 JXL 取代;zstd 压图很弱) |
| **Chinchilla** | LLM | **引用 Delétang(ImageNet)** | G2 | ❌引用 | **P0** |
| **P2LLM** | LLM(NeurIPS'25,image SOTA) | 引用其论文 | (其数据) | 优先引用 | P1 |
| OmniZip | 多模态 | 引用(CLIC-M) | G1(CLIC) | 优先引用 | P1 |
| **前作 = 你的 static** | bGPT | 你的 repo(`--mode static`) | G1+G2 | ✅内置 | **P0** |
| **Ours (online)** | bGPT | 你的 repo | G1+G2 | ✅你的方法 | **P0** |

## 5. AUDIO

| Baseline | 类型 | 来源 | 数据 | 重现? | 优先级 |
|---|---|---|---|---|---|
| **FLAC** | 传统 · 事实标准 | `flac --best` | G1 | 跑工具 | **P0** |
| **OptimFROG** | 传统 · **压缩率 SOTA** | `ofr --preset max` | G1 | 跑工具 | **P0** |
| ~~WavPack / APE / ALAC~~ | 冗余 | — | — | — | ✂️ 删(FLAC+OptimFROG 已够) |
| **Chinchilla** | LLM | **引用 Delétang(LibriSpeech)** | G2 | ❌引用 | **P0** |
| Llama3-8B* | LLM | base 论文实现(可复现) | G1 | 你的 static 变体 | P1 |
| OmniZip | 多模态 | 引用(LibriSpeech) | G2 | 引用 | P1 |
| **前作 = 你的 static** | bGPT | 你的 repo | G1+G2 | ✅内置 | **P0** |
| **Ours (online)** | bGPT | 你的 repo | G1+G2 | ✅你的方法 | **P0** |

> ⚠️ **audio 的 bit-depth 对齐坑**：你的方法在 **8kHz/8-bit/mono PCM** 上工作；FLAC/OptimFROG 惯例在 16-bit 上跑。**ratio 分母必须一致**——把传统 codec 也喂**同一份 8kHz/8-bit PCM**（FLAC 支持 8-bit；OptimFROG 需确认），或全部相对同一 canonical PCM 报 bps。否则数字不可比。base 论文 audio 相对原始文件字节算 ratio——确认你与它对齐。

---

## 5b. 传统 SOTA 定位核实 + License（每个声称都有依据）

| 模态 | baseline | 是 SOTA 吗 | 依据 | License / 可用性 |
|---|---|---|---|---|
| text | xz/LZMA2 | ❌ 否,是"最强**实用**通用"(≈2 阶 Shannon) | 传统极限是 context-mixing 的 **cmix/paq8**(Mahoney LTCB enwik9 榜);xz far behind | 0BSD/公共领域,自由 ✓ |
| text | cmix/paq8 | ✅ 传统极限 SOTA | Large Text Compression Benchmark 榜首级 | GPL(可用;**极慢+enwik 专用调优**,建议仅文字提及、不跑) |
| image | **JPEG-XL** | ✅ **传统(非神经)SOTA** | ISO CE8.2 各类别最密;比任何 web codec 小 45%;FLIF 已被其取代 | BSD-3(libjxl),自由 ✓ |
| audio | **OptimFROG** | ✅ **传统压缩率 SOTA** | 多来源 "best-ratio lossless codec" | ⚠️专有闭源 freeware:report benchmark OK,**勿打包重分发** |

**含神经/LLM 的 SOTA（所有模态一致）：** LLM 压缩(Chinchilla/LMCompress/**你**)早已**全面超越所有传统**(text rate ~8% vs 传统最好 ~15%)。→ **你真正的对手是其他 LLM 压缩(Chinchilla/P2LLM/OmniZip)，传统 baseline 只是锚点**，它谁最强不影响你的定位。

**侵权澄清：** 在论文里**报告"用某工具压缩数据的压缩率"从不构成侵权**——所有压缩论文的标准做法。你只在自己机器跑、报数、不重分发工具。自由开源(gzip/brotli/xz/PNG/WebP/JPEG-XL/FLAC)连打包都无忧;OptimFROG 闭源但免费、report 学术界普遍接受;cmix/paq 是 GPL,报数不受传染性影响。

## 6. Action checklist（按优先级,你就照这个顺序测）

**P0 — 没有它论文不成立（先做）**
- [ ] 你的 **static + online** 跑遍 G1（headline 主结果 + Δ%，你已在做）
- [ ] 你的 **static + online** 跑 G2（enwik9 / ImageNet32 / LibriSpeech）← 为了和引用数字并列
- [ ] 传统 P0：text{gzip, brotli, xz/LZMA2} · image{PNG, WebP, JPEG-XL} · audio{FLAC, OptimFROG}，在 G1+G2 上跑（每模态 = 锚点 + 现代 + SOTA，无冗余无争议）
- [ ] 抓 **Chinchilla** 数字填表（已整理好，见 §7 数值）

**P1 — 让表更有分量**
- [ ] 引用 **P2LLM / OmniZip** 数字（image/多模态）
- [ ] model-scaling 曲线（0.6/1.7/4/8B，追赶叙事）

**P2 — 有余力再补**
- [ ] 引用 FineZip/LLMZip 等

## 7. 已备好的引用数值（Chinchilla, raw, chunked 2048B, 1GB）

| Chinchilla | text(enwik9) ratio | image(ImageNet) ratio | audio(LibriSpeech) ratio |
|---|---|---|---|
| 1B | 8.85 | 1.61 | 4.02 |
| 7B | 9.80 | 1.83 | 4.24 |
| 70B | 12.05 | 2.08 | 4.76 |

（rate%: 1B 11.3/62.2/24.9 · 7B 10.2/54.7/23.6 · 70B 8.3/48.0/21.0；bpb = rate×0.08。Llama2-7B: 11.24/1.87/4.33。）

## 8. 主表模板（每模态一张，填进论文）

| Method | Model | Size | ratio↑ / bpb↓ | 传 model? | Δ(static→online) |
|---|---|---|---|---|---|
| gzip / brotli / FLAC / JPEG-XL … | – | – | (自跑) | – | – |
| Chinchilla 7B / 70B | Chinchilla | 7/70B | (引用) | raw | – |
| P2LLM / OmniZip | (其模型) | – | (引用) | – | – |
| LMCompress(前作) = **static** | Qwen3-x / bGPT | x B | (自跑) | no | – |
| **Ours (online)** | Qwen3-x / bGPT | x B | (自跑) | **no (0 bit)** | **+Δ%** |

外加三张核心图：① Δ vs 数据量（amortization）② static/online rate vs model size（追赶）③ Δ vs 数据同源度（增益来源）。

---

## 附录 A · 传统工具命令（lossless，最强设置）

```bash
# --- text (对整文件 = unchunked；chunked 版按 2048B 切后逐块压再求和) ---
gzip -9 -k f.txt              # .gz   锚点
brotli -q 11 f.txt            # .br   现代 web (Google)
xz -9e -k f.txt               # .xz   LZMA2, 最强通用 (Chinchilla 用)

# --- image (先转 canonical RGB，例如 PNG/BMP；ratio 分母 = W*H*3) ---
optipng -o7 img.png                              # PNG      锚点
cwebp -lossless -q 100 -m 6 img.png -o img.webp  # WebP     现代 (Google)
cjxl img.png img.jxl -d 0 -e 9                   # JPEG-XL  非神经 SOTA (d0=lossless)

# --- audio (⚠️ 与你的方法同一份 8kHz/8-bit/mono PCM) ---
flac --best in.wav                     # FLAC       事实标准
ofr --preset max in.wav                # OptimFROG  压缩率 SOTA
```

比率一律 `ratio = 原始canonical字节 / 压缩后字节`；跨模态比用 ratio，同格式内可比 bpb。
