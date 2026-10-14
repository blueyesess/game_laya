# mario-laya — 让 laya 玩超级马里奥

把 [laya](https://github.com/NandhaKishorM/laya) 当成马里奥的**反射神经**：NES 模拟器
实时跑，画面上的局面被翻译成一段文本喂给 laya，laya 用它的三种类型化决策头
(`choice` / `score` / `noul`) 吐出「往哪走 / 要不要跳 / 危险多大」，再翻译回手柄按键。
一个网页看板把这套闭环整个摊开给你看。

> 先说清楚一件事：laya **不是**游戏引擎。它是一个 Python 包
> （`pip install laya`），用 ModernBERT / mmBERT 在单次前向里对任意文本做类型化决策，
> 附带一个按语种挑 checkpoint 的 router。它没有渲染、没有精灵、没有物理。
> 所以这里的分工是：**nes_py 负责画面，laya 负责「在想什么」。**

---

## 架构

```
                 ┌──────────────── 60.1 Hz，模拟器线程 ────────────────┐
  roms/smb1.nes ─┤ nes_py.LaiNES                                     │
                 │   ├─ frame  240×256×3 ─────────────┐               │
                 │   └─ RAM    2 KB ──┐                │               │
                 └────────────────────┼────────────────┼───────────────┘
                                      │                │
                    smb_state.decode()│                │ frame_to_rgb_bytes()
                                      ▼                ▼
                              SmbState  ◄──────────── raw RGB ──────► 看板 canvas
                                 │
                          scene.build_state()
                                 │  dict → laya 序列化
                                 ▼
                 ┌──────────────── 3–7 Hz，脑线程 ──────────────────────┐
                 │ LayaBrain.decide(state)                            │
                 │   move  : choice{right,left,wait}  → ←/→           │
                 │   jump  : noul  P(true) > 0.5      → A             │
                 │   risk  : score{0,1,2} 序数          → 辅助/仲裁      │
                 │   answer_confidence < 下限 → 交给 reflex.py         │
                 └──────────────────────┬──────────────────────────────┘
                                        │  持有上一帧决策
                                        └──────► 模拟器线程持续按住的按键
```

**两个线程，谁也不等谁。** 这是整个设计的出发点：NES 一秒 60.0988 帧，一个 322M
编码器跑三个问题的前向大约 145 ms。所以模拟器全速自由运行，脑线程按它自己的速度
决策，**两次决策之间一直保持上一次的按键**——这恰好就是真实手柄输入的行为，而 SMB1
本来就是为这种玩法设计的。

---

## 快速开始

```bash
# 1. 基础层：模拟器 + 看板 + 规则层策略（不需要模型）
.venv\Scripts\python -m pip install -e .

# 2. brain 层：让 laya 真正参与决策
#    torch 必须先单独从 CPU 索引装 —— PyPI 的 Windows 构建是 CUDA，约 2.5GB
.venv\Scripts\python -m pip install torch --index-url https://mirrors.aliyun.com/pytorch-wheels/cpu/
.venv\Scripts\python -m pip install -e ".[brain]"

# 3. 放 ROM（本作不附带、不下载任何受版权保护的 ROM）
#    roms\smb1.nes —— 用你自己合法获取的 dump

# 4. 先用自制 homebrew ROM 验证整条链路（不需要 SMB1）
.venv\Scripts\python tools\make_test_rom.py
.venv\Scripts\python tools\smoke_test.py

# 5. 跑起来
.venv\Scripts\python run.py --rom roms\smb1.nes --policy hybrid --warmup
#    打开 http://127.0.0.1:8770/
```

依赖只有一层时（第 1 步），看板和 `reflex` 策略完全可用，`laya` / `hybrid` 策略会返回
一句明确的安装提示而不是崩溃。

**本机 pip 需要显式代理**，否则索引拉不到，会误报成
`Could not find a version that satisfies the requirement numpy (from versions: none)`：

```bash
.venv\Scripts\python -m pip install --proxy http://127.0.0.1:57417 --no-cache-dir <包名>
```

依赖清单在 `pyproject.toml`，不在 `requirements.txt`。

`uv.lock` 也一并提交：它是当前环境的完整解析结果（含 brain extra 里的 laya 依赖链），
有它在别人才能装出同一套版本。注意本机 uv 的命令本身不可用，这个 lock 是安装时
生成的产物而非日常使用的工具；实际安装按上面的 pip 命令走即可。

首次运行会从 HuggingFace 下载 checkpoint（`multilingual` 约 1.2 GB，`english` 约 1.7 GB）。
`--warmup` 会在开服前就把权重加载好，免得开局前几十秒马里奥在原地发呆。

---

## 看板上能看到什么

| 面板 | 内容 |
|---|---|
| 画面 | nes_py 的原始帧，黄线是识别到的马里奥，红块是检测到的地面缺口 |
| 喂给 laya 的观测 | 实际送进模型的那段文字（`scene`）+ 结构化字段（金币/时间/最近敌距/是否塌方） |
| laya 的决策 | `move` 的完整概率分布、`jump` 的 P(true) 刻度条、`risk` 的序数分布、`answer_confidence`、路由到的 checkpoint、延迟 ms |
| 决策流水 | 最近 40 次决策，每行标注来源是 `laya` 还是 `reflex` |
| 控制 | 切换 policy / 手动接管（方向键、Z=B、X=A、Enter=START）/ 调置信度下限与模拟速度 |

黄色竖线和红色缺口块是**故意画出来的自检**：如果它们和画面里马里奥的真实位置对不上，
说明 `smb_state.find_mario()` 的颜色阈值或 `ground_profile()` 需要调，一眼就能看出来。

---

## 三种策略

```bash
python run.py --policy laya     # 只用模型。诚实，但模型一卡马里奥就站着不动
python run.py --policy reflex   # 只用规则。作为对照组，回答「模型到底有没有用」
python run.py --policy hybrid   # 默认：置信度够就听模型，不够就交给规则层
```

`reflex.py` 不是装饰。laya 几秒才决策一次，而 NES 一秒跑六十次，中间总得有人按着
按键；`answer_confidence` 低于下限时模型实际上在说「我不知道」，这时候执行一个随机
选择比执行一条确定的规则更糟。**看板会给每一次决策标注来源，hybrid 模式不会假装
模型做了规则层做的事。**

---

## 关于 RAM 映射的诚实说明

`src/config.py` 里的 SMB1 内存映射来自社区文档。**计时器 ($0021)、世界 ($0025)、
生命 ($0026)、状态 ($0027) 和玩家坐标 ($0028/$0029) 是各家地图一致的、可以直接信的部分。**
敌人槽位 ($0030–$0056) 各家表格之间存在字节级差异，我没有在真机 ROM 上验证过。
所以配套提供了标定工具：

```bash
.venv\Scripts\python tools\dump_ram.py --rom roms\smb1.nes --script "wait*90,right*300"
```

它做两个**不依赖任何单个地址正确**的交叉验证：`world_x` 来自 RAM，`mario_screen_x`
来自在画面里找马里奥那两个精确颜色，两者独立；按右键时它们应当同步增长。另外还会
dump 完整 2 KB 十六进制视图，方便一眼看出哪个地址明显不对，然后在 `src/config.py`
里改一行即可。

---

## 没有 ROM 也能证明什么

`tools/make_test_rom.py` 用一个 60 行的迷你 6502 汇编器**现场生成一个合法 homebrew ROM**：
每个 vblank 都去读手柄，把 8 个按键存进 RAM $0301，再把调色板 $3F00 写成
「右键=红 / 左键=绿 / A=白 / 无=蓝」。

于是有两件事变成了真正的断言（见 `tools/smoke_test.py`）：

* 6502 读手柄的顺序是 A,B,Select,Start,Up,Down,Left,Right，正好等于 nes_py 的
  **bit0=A … bit7=RIGHT**。所以从 Python 读回 $0300，就等于**拿模拟器本身**校验了
  `config.py` 里的按键位表，而不是校验我的记忆。
* 画面颜色随按键变化 → 视频通路可验证，而且用眼睛就能看出来。

`tools/smoke_test.py` 还会核对 `buttons_to_byte()` 与 `env.get_keys_to_action()`
在全部 256 种组合下完全一致。

---

## 文件

```
run.py                    入口
src/config.py             按键位表、SMB1 RAM 映射、调色板、可调参数
src/emulator.py           nes_py.NESEnv 的防御性封装 + 帧率节拍器
src/smb_state.py          RAM + 帧缓冲 → SmbState（马里奥定位、地面剖面、敌人）
src/scene.py              SmbState → laya 的 state（dict，含一句话描述）
src/brain.py              laya Router + 三个问题的 schema + 答案归一化
src/reflex.py             确定性兜底策略
src/agent.py              模拟器线程 + 脑线程 + 决策日志
src/server.py             Flask：原始帧流 / 快照 / 决策流水 / 控制接口
web/index.html            看板
tools/make_test_rom.py    迷你 6502 汇编器 + 测试 ROM 生成
tools/verify_rom.py       只用标准库校验 ROM 结构
tools/smoke_test.py       端到端冒烟测试（16 项）
tools/probe_emulator.py   分层探针：cpu / ctrl / ppu / nmi
tools/probe_controller.py 测定 $4016 的精确语义
tools/probe_cartridge.py  问 C++ 内核为什么拒绝某个 ROM
tools/probe_screen.py     对比 nes_py 两种取帧方式，验证 /api/frame.raw 的字节契约
tools/inspect_nespy.py    dump 已安装 nes_py 的真实 API
tools/check_dashboard.py  看板静态校验：node --check 解析 JS、核对 id 与端点（13 项）
tools/blit_check.js       从 index.html 抠出 blit() 并在 node 里跑
tools/check_blit.py       真实帧走完 模拟器→HTTP→canvas（4 项）
tools/dashboard_selftest.py  进程内验证看板全部路由（16 项）
tools/check_deps.py       pyproject 声明与实际 import 双向一致（6 项）
tools/find_ram.py         用像素真值反查 RAM 地址
tools/verify_ram.py       用「计时器递减」「死一次掉一条命」判决 RAM 映射
tools/probe_palette.py    自动定位马里奥并列出他身上的实际颜色
tools/probe_ground.py     单测背景/地面剖面，打印中间值
tools/run_headless.py     无服务无浏览器跑一段，按采样输出状态表 + 逐帧 PNG
tools/bench_brain.py      实测 laya 延迟 / input_tokens / answer_confidence
tools/dump_ram.py         SMB1 RAM 映射标定（早期版本，功能被 find_ram 覆盖）
logs/decisions.jsonl      每次决策的完整输入输出
```

## 测试现状

五套测试都不需要 SMB1，全部通过：

```
tools/smoke_test.py         16/16   模拟器、位表、按键往返、解码、观测契约、Flask 路由
tools/dashboard_selftest.py 16/16   全部路由、控制指令、线程存活、决策速率
tools/check_dashboard.py    13/13   JS 语法、DOM id、端点一致性、canvas 尺寸、RGB→RGBA
tools/check_blit.py          4/4    真实帧走完 模拟器→HTTP→canvas，逐字节等于标准 RGBA
tools/check_deps.py          6/6    pyproject 声明与实际 import 双向一致
tools/probe_emulator.py     cpu / ctrl / ppu / nmi 全 PASS
```

`tools/probe_emulator.py` 存在的理由是它抓到了一个**静默**的 bug：iNES 头第 5 字节的
单位是 16 KB，我最初给 32 KB 的 PRG 写了 `1`，于是内核只加载前一半，
`$FFFA` 的向量落在 16 KB 映像的 `$3FFA`——那里全是 0，CPU 复位到 `$0000`
执行一串 `BRK`，看起来像「模拟器没跑」。这种错误不会有任何报错。

`tools/check_dashboard.py` + `tools/check_blit.py` 抓的是另一类**同样静默**的 bug：
把 24 位 RGB 流直接塞进 32 位 RGBA 的 `ImageData`，每个像素错位一字节，画面变成
逐行递增的彩色斜纹——而 HUD、砖块、地面都还在正确位置，看起来像模拟器坏了。
`check_blit.py` 不复述 blit 逻辑，而是用正则从 `web/index.html` 里**抠出真正的那段
代码**丢给 node 跑，再和 Python 侧算的标准 RGBA 展开逐字节比对。

## 三个已定位的问题

### 画面斜纹（已修）
`ctx.createImageData()` 是 RGBA、4 字节/像素 = 245760 字节；NES 帧是 RGB、
3 字节/像素 = 184320 字节。直接 `image.data.set(rgb)` 会让每个像素错位一字节。
`alpha:false` 只是让 canvas 按不透明合成，不会修正 RGB 错位。

### 这份 ROM 的 RAM 映射与社区表不符（已降级，未解决）
`25th Anniversary (Europe) (Promo, Virtual Console)` 上，`$0025`（世界）、`$0026`（生命）、
`$0027`（状态）、`$0028/$0029`（坐标）在整关内**全为 0**。这不是小偏差，是整张表不适用。

后果比「看板上数字不对」严重：这些数字会被序列化进 laya 读的文本，等于告诉模型
「world 0-1，0 条命，0 时间」。所以 `smb_state.ram_is_plausible()` 用**游戏本身的不变量**
（世界从 1 开始、有生命、金币 0-99、计时器 ≤400、状态是七个已知值之一）校验，
不通过就把观测降级成纯像素信息，并明说「未知」而不是报 0。
敌人槽位同样会被清空——错的映射不会给你 0 个敌人，而是塞着别的游戏数据，
拿 world_x=0 去算距离会凭空造出敌人。

标定工具已经就位，用像素识别出的马里奥位置当真值反查：

```
python tools/find_ram.py  --rom <rom>    # 找与马里奥同步变化的字节
python tools/verify_ram.py --rom <rom>    # 用「计时器递减」「死一次掉一条命」做判决
```

两者都还没能在这份 ROM 上给出结论，因为 `find_mario()` 当时会在**标题画面**误报
（见下条），导致分段全是噪声。

### 视觉层连着错了三处（都已修）
这三处都是**静默**的：照常返回数字、照常渲染，只是返回的是错的。

**1. 地面剖面 off-by-32。** `ground_profile()` 里
`lowest = SCREEN_H - 1 - argmax(...)` 用的是 **239**，但裁掉 32 行 HUD 之后
playfield 只有 **208** 行。于是每个真实地面行都被**下移 32**，正好落到 ≥240 ——
而 ≥240 在协议里是「坑」的哨兵值。结果每一列都被读成空洞，`on_ground` 和
`pit_ahead` 一直是假的。同一函数里天空色写死 `#5C94FC` 也是这类问题：
这份 ROM 的天空是 `#6888FC`，容差 8 判不出来，于是整片天空被当成实心地面。
现在背景色与地面色都改为**从画面自身推断**（playfield 顶部/底部占比最高的颜色）。

**2. find_mario 错了两版。** 纯红阈值（`$D82800`，马里奥的帽子）和版权画面的
`(c)1985 NINTENDO` 同色，于是在标题画面 frame 34 就报「看到马里奥」，
把上层所有标定分析带偏。改成「先用纯蓝锚定」（overall 是全游戏唯一用 `$0000FC`
的东西）修好了标题画面，但这份 ROM **既没有 `#D82800` 也没有纯蓝**——
它的马里奥红是 `#F83800`。于是 `mario_screen_x` 连续 2000 多帧为 None，
而画面上马里奥就站在那儿。

第三版不写死任何颜色，只用**几何 + 相对配色**：背景 = playfield 顶部占比最高的颜色，
地面 = 底部占比最高的颜色，两者减掉后剩下的就是物体，再取「站在地面上、最靠左的
紧凑色块」。这是 SMB1 各版本都成立的不变量。35 秒 headless 实测
**136/138 帧检测到马里奥，x 从 24 走到 1110，137 个不同值**——确实在跟着人跑。

局限说清楚：「地面上最靠左的物体」只在关卡开头是马里奥。一旦 Goomba 跑到他左边、
或他钻到砖块下面，就会认错。这是起点不是 solved——也正是 `ram_plausible` 存在的理由。

### START 恢复有两个独立的 bug（都已修）
**1. 一次性预算。** 原本是「`frame < 900` 之前一直按 START」，越过上限就**永久失效**。
真实运行里出现过：第 8000 帧，WORLD 1-1 的 HUD、Goomba、砖块、地面全都在画面上，
标题卡片却盖在最上面出不去——因为再没有任何按键到得了控制台。

**2. 长按不算按键。** 改成条件判断后仍然失败。SMB1 标题画面用**边沿检测**，
而「标题 → 1 PLAYER GAME → 关卡卡片」需要**两次独立按键**。实测 45 秒、2400 帧
一直按住 START，马里奥一次都没出现过。START 必须**脉冲**：24 帧周期里按住 6 帧。

触发条件用「马里奥已连续 24 帧不在画面上」，而不是「开机后 900 帧内」——
前者对开机、死亡、Game Over 三种情况都成立，不需要特例分支。
START 在正常游戏过程中无副作用，所以偶尔误判为真也没有代价。
`tools/dashboard_selftest.py` 用**两个分开的断言**锁住脉冲：既要看到按下相位，
也要看到松开相位——单张快照区分不了这两种失败。

### 推理延迟：3.5 秒/次，以及一个真实的权衡
`tools/bench_brain.py` 在本机实测（CPU，multilingual 322M，3 个问题）：

| state 长度 | input_tokens | 中位延迟 | move 概率分布 | answer_confidence |
|---|---|---|---|---|
| 长散文 + 完整 criteria | 926 | 5068 ms | 0.875 / 0.057 / 0.068 | 0.607 |
| 短 state + 极简 criteria | 368 | 2693 ms | 0.382 / 0.311 / 0.309 | 0.382 |
| 短 state + 完整 criteria（当前） | 429 | 3499 ms | 0.440 / 0.311 / 0.249 | 0.440 |

关键机制：**laya 会把 state 按 question 重复编码**，所以 state 越长，每个问题付三倍。
把冗余的结构化字段去掉（散文本来就在重述它）是纯赚，926→368 换 5068→2693 ms。

但把 criteria 写短是**有损**的：模型没有依据区分三个选项，答案趋近均匀。
而且置信度也吃 state 长度——恢复 criteria 只把 0.382 拉回 0.440，回不到 0.607。
**没有免费午餐，这是真实权衡**，当前停在中间档。

置信度下限因此从 0.45 调到 0.40：实测典型值就是 0.44，下限设 0.45 会把几乎
每一次决策都交给规则层，hybrid 等于退化成纯 reflex。

0.29 Hz 意味着**每 210 个 NES 帧才一次决策**。架构上靠「降频决策 + 保持上次按键」
撑住，这正是当初把两个线程解耦的原因。

---

## 已知限制

- **`score` 是 laya 自己文档里说的最弱原语。** `risk` 因此只用来辅助：它不直接驱动
  按键，只在 `move`/`jump` 都不确定时参与仲裁。
- **决策率上不去。** 322M 模型在 CPU 上就是几 Hz。这不是工程没做好，是物理限制，
  所以架构选择是「降频决策 + 保持输入」而不是「每帧决策」。
- **敌人种类分不出来。** SMB1 的 RAM 里不存敌人类型。看板显示的是位置（精确），
  种类只是占位。
- **`find_mario()` 只在真 SMB1 画面上验证过。** 测试 ROM 是纯色，`decode()` 在它上面
  会把整屏中心误判成马里奥（`mario_screen_x=127`）。真游戏里没有这个问题，但说明
  颜色阈值需要用真 ROM 复核——`tools/dump_ram.py` 的交叉验证就是为此准备的。
- **laya 的 checkpoint 首次运行要下 1.2–1.7 GB。** 本机网络不稳，pip 都能断十几次，
  这个步骤很可能需要重试或手动 `hf download`。
