# nes-laya — 让 laya 玩任意 NES 游戏

把 [laya](https://github.com/NandhaKishorM/laya) 当成 NES 玩家的**反射神经**：模拟器实时跑，
**画面被直接翻译成结构化观测**喂给 laya，laya 用它的三种类型化决策头
(`choice` / `score` / `noul`) 吐出「往哪走 / 要不要跳 / 危险多大」，再翻译回手柄按键。
一个网页看板把这套闭环整个摊开给你看。

> 先说清楚一件事：laya **不是**游戏引擎。它是一个 Python 包
> （`pip install laya`），用 ModernBERT / mmBERT 在单次前向里对任意文本做类型化决策，
> 附带一个按语种挑 checkpoint 的 router。它没有渲染、没有精灵、没有物理。
> 所以这里的分工是：**nes_py 负责画面，laya 负责「在想什么」。**

**这是通用项目，不是马里奥专用。** 游戏相关的东西全部搬进了
`src/profiles/*.json`——一个 JSON 文件描述一个游戏，不改一行 Python 就能驱动另一款 ROM。

---

## 架构

```
                 ┌──────────────── 60.1 Hz，模拟器线程 ────────────────┐
  roms/*.nes  ───┤ nes_py.LaiNES                                     │
                 │   ├─ frame  240×256×3 ─────────────┐               │
                 │   └─ RAM    2 KB ──┐                │               │
                 └────────────────────┼────────────────┼───────────────┘
                                      │                │
                  profile.ram 可选读取 │                │ frame_to_rgb_bytes()
                                      ▼                ▼
                  ┌──────────────────────────┐    raw RGB ──────► 看板 canvas
                  │ src/vision.py（通用，无游戏知识）│
                  │   背景色 / 地面剖面 / 连通域     │
                  │   角色定位 / 坑 / 墙 / 可攀爬   │
                  └────────────┬─────────────┘
                               │
                  src/state.py + src/profiles/<game>.json
                               │  GameState → scene.build_state()
                               ▼
                 ┌──────────────── 3–7 Hz，脑线程 ──────────────────────┐
                 │ LayaBrain.decide(state)                            │
                 │   move  : choice{progress,back,wait} → 画像里的按键 │
                 │   jump  : noul  P(true) > 0.5      → jump 按键      │
                 │   risk  : score{0,1,2} 序数          → 辅助/仲裁      │
                 │   answer_confidence < 下限 → 交给 reflex.py         │
                 └──────────────────────┬──────────────────────────────┘
                                        │  持有上一帧决策
                                        └──────► 模拟器线程持续按住的按键
```

**两个线程，谁也不等谁。** 这是整个设计的出发点：NES 一秒 60.0988 帧，一个 322M
编码器跑三个问题的前向大约 145 ms。所以模拟器全速自由运行，脑线程按它自己的速度
决策，**两次决策之间一直保持上一次的按键**——这恰好就是真实手柄输入的行为，
而 NES 游戏本来就是为这种玩法设计的。

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
#    roms\你的游戏.nes，然后 --rom 显式指定

# 4. 先用自制 homebrew ROM 验证整条链路
.venv\Scripts\python tools\make_test_rom.py
.venv\Scripts\python tools\smoke_test.py

# 5. 跑起来（--profile 缺省是 generic，任何 NES 横版都能用）
.venv\Scripts\python run.py --rom roms\你的游戏.nes --profile generic --policy hybrid
#    有 SMB1 ROM 的话，smb1 画像还能额外读 RAM 给出金币/时间/生命
.venv\Scripts\python run.py --rom roms\smb1.nes --profile smb1 --policy hybrid --warmup
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
有它在别人才能装出同一套版本。

首次运行会从 HuggingFace 下载 checkpoint（`multilingual` 约 1.2 GB，`english` 约 1.7 GB）。
`--warmup` 会在开服前就把权重加载好，免得开局前几十秒角色在原地发呆。

---

## 加一个新游戏 = 写一个 JSON

`src/profiles/generic.json` 是起点，直接复制一份改：

| 字段 | 作用 |
|---|---|
| `hud_rows` | HUD/状态栏占掉顶部多少行；playfield 从它下面开始 |
| `sprite_h` | 角色精灵大概多高（像素），用来过滤「这东西不像人」 |
| `lookahead_px` | 往前看多少像素找坑和墙 |
| `climb_lookahead_px` / `max_wall_rise` | 可攀爬判定的窗口与最小落差 |
| `goal` | 一句话目标，会写进喂给 laya 的观测里 |
| `move` | 决策词汇表 → 实际按键。`{"progress":["right"],"back":["left"]}` |
| `actions` | 具名动作 → 按键。`{"jump":["A"],"run":["B"],"up":["up"]}` |
| `jump_hold_frames` | 跳跃键按住几帧（NES 的跳跃弧线是固定的，单帧点按跳不起来） |
| `ram` | **可选**。社区记录的内存布局；不填就纯靠像素，一样能玩 |

`smb1.json` 在 generic 之上多填了 `ram` 与 `invariants`，所以看板上多出金币/时间/生命/
敌人位置。`generic` 不读 RAM，这些字段明确显示「不可用」而不是显示 0——见下文。

---

## 看板上能看到什么

| 面板 | 内容 |
|---|---|
| 画面 | nes_py 的原始帧 + **视觉层输出的叠加层**：黄线/黄框是识别到的角色，红块是坑，橙块是墙，绿框是其它物体 |
| 喂给 laya 的观测 | 实际送进模型的那段文字（`scene`）+ 结构化字段（坑/墙/可攀爬/HUD） |
| laya 的决策 | `move` 的完整概率分布、`jump` 的 P(true) 刻度条、`risk` 的序数分布、`answer_confidence`、路由到的 checkpoint、延迟 ms |
| 决策流水 | 最近 40 次决策，每行标注来源是 `laya` 还是 `reflex` |
| 控制 | 切换 policy / 手动接管（方向键、Z=B、X=A、Enter=START）/ 调置信度下限与模拟速度 |

叠加层是**故意画出来的自检**：如果黄框和画面里角色的真实位置对不上，
说明 `vision.find_character()` 判据需要调，一眼就能看出来。

---

## 三种策略

```bash
python run.py --policy laya     # 只用模型。诚实，但模型一卡角色就站着不动
python run.py --policy reflex   # 只用规则。作为对照组，回答「模型到底有没有用」
python run.py --policy hybrid   # 默认：置信度够就听模型，不够就交给规则层
```

`reflex.py` 不是装饰。laya 几秒才决策一次，而 NES 一秒跑六十次，中间总得有人按着
按键；`answer_confidence` 低于下限时模型实际上在说「我不知道」，这时候执行一个随机
选择比执行一条确定的规则更糟。**看板会给每一次决策标注来源，hybrid 模式不会假装
模型做了规则层做的事。**

---

## 视觉层：只用几何，不用颜色

`src/vision.py` 一行颜色阈值都没有。原因是硬编码颜色在真实 ROM 上连着错了两版：

* 纯红 `$D82800`（马里奥的帽子）和版权画面的 `(c)1985 NINTENDO` 同色，于是在标题画面
  frame 34 就报「看到马里奥」，把上层所有标定分析带偏；
* 改成「先用纯蓝锚定」修好了标题画面，但 Virtual Console 重制版**两个颜色都没有**，
  它的马里奥红是 `#F83800`。于是位置连续 2000 多帧为 None，而画面上人就站在那儿。

同理，天空色写死 `#5C94FC` 也静默失败：这份 ROM 的天空是 `#6888FC`，容差 8 判不出来，
于是整片天空被当成实心地面，每一列都读成坑。现在背景色、地面色、地面剖面**全部
从画面自身推断**，这是横版 NES 都成立的不变量。

角色定位的三个几何判据：

1. 角色站在地面上——底边在它自己那一列地表的一个精灵高度内（用**局部**地表，
   不是全屏中位数，否则爬上楼梯就被永久判成腾空）；
2. 画面里有第二个物体时，角色是离屏幕中间三分之一最近的那个；
3. 只有一个候选时（关卡刚开场），报 `uncertain=True`——通常是角色，但说清楚是猜的，
   因为模型会基于错误位置「自信地错」，那是任何置信度阈值都拦不住的。

再加一个**逐帧记忆的包围盒**当先验：它比任何几何规则都更能扛住「一只 Goomba 跑到
角色左边」——那正是旧版「取地面上最靠左的物体」规则翻车的地方。

**诚实的局限**：这套判据是横版过关的启发式。垂直卷轴游戏（魂斗罗式）、俯视游戏
（塞尔达式）需要新的画像和新的视觉判据，不是改几个数字的事。`tools/run_headless.py`
跑一段并逐帧存 PNG，就是用来一眼看出识别对不对的。

---

## 关于 RAM 映射的诚实说明

`src/profiles/smb1.json` 的内存映射来自社区文档。计时器 (`$0021`)、世界 (`$0025`)、
生命 (`$0026`)、状态 (`$0027`) 和玩家坐标 (`$0028/$0029`) 是各家地图一致的、可以直接信的。
敌人槽位 (`$0030–$0056`) 各家表格之间存在字节级差异，我没有在真机 ROM 上验证过。

关键是**错了会怎样**：错的映射不会给你 0，它会给你**看似合理的假数据**。某个
25 周年纪念版上这些地址在整关内全为 0，于是观测会变成「world 0-1，0 条命，0 时间」
——而马里奥正在画面上跑。模型会照着这个虚构推理。

所以 `RamMap.plausible()` 用**游戏本身的不变量**校验（世界从 1 开始、有生命、金币 0-99、
计时器 ≤400、状态是已知值之一），不通过就把观测降级成纯像素信息，并明说「未知」
而不是报 0。`generic` 画像没有 RAM 表，`ram_plausible` 恒为 False，天然走这条路。

配套标定工具：

```bash
python tools/smb1/find_ram.py   --rom roms/smb1.nes   # 用像素真值反查 RAM 地址
python tools/smb1/verify_ram.py --rom roms/smb1.nes   # 用「计时器递减」「死一次掉命」判决
python tools/smb1/dump_ram.py   --rom roms/smb1.nes   # 全 2KB hexdump + RAM/像素一致性交叉验证
```

---

## 没有 ROM 也能证明什么

`tools/make_test_rom.py` 用一个 60 行的迷你 6502 汇编器**现场生成一个合法 homebrew ROM**：
每个 vblank 都去读手柄，把 8 个按键存进 RAM `$0301`，再把调色板 `$3F00` 写成
「右键=红 / 左键=绿 / A=白 / 无=蓝」。

于是有两件事变成了真正的断言（见 `tools/smoke_test.py`）：

* 6502 读手柄的顺序是 A,B,Select,Start,Up,Down,Left,Right，正好等于 nes_py 的
  **bit0=A … bit7=RIGHT**。所以从 Python 读回 `$0300`，就等于**拿模拟器本身**校验了
  `config.py` 里的按键位表，而不是校验我的记忆。
* 画面颜色随按键变化 → 视频通路可验证，而且用眼睛就能看出来。

`tools/smoke_test.py` 还会核对 `buttons_to_byte()` 与 `env.get_keys_to_action()`
在全部 256 种组合下完全一致，以及画像驱动的观测/问法/规则层产出的按键都是 NES 认识的。

---

## 文件

```
run.py                       入口
src/config.py                按键位表、屏幕尺寸、可调参数（不再有任何游戏知识）
src/vision.py                通用视觉层：地面剖面、连通域、角色定位、坑/墙/攀爬
src/state.py                 GameProfile + GameState + 画像驱动的 Decoder
src/profiles/generic.json    零配置画像，任何横版 NES
src/profiles/smb1.json       SMB1：generic 之上加 RAM 表与不变量
src/scene.py                 GameState → laya 的 state（dict，含一句话描述）
src/brain.py                 laya Router + 由画像生成的问法 + 答案归一化
src/reflex.py                确定性兜底策略（只读视觉层能证明的东西）
src/agent.py                 模拟器线程 + 脑线程 + 决策日志
src/server.py                Flask：原始帧流 / 快照 / 决策流水 / 控制接口
web/index.html               看板
tools/make_test_rom.py       迷你 6502 汇编器 + 测试 ROM 生成
tools/verify_rom.py          只用标准库校验 ROM 结构
tools/smoke_test.py          端到端冒烟测试（20 项）
tools/probe_emulator.py      分层探针：cpu / ctrl / ppu / nmi
tools/probe_controller.py    测定 $4016 的精确语义
tools/probe_cartridge.py     问 C++ 内核为什么拒绝某个 ROM
tools/probe_screen.py        对比 nes_py 两种取帧方式，验证 /api/frame.raw 的字节契约
tools/inspect_nespy.py       dump 已安装 nes_py 的真实 API
tools/check_dashboard.py     看板静态校验：node --check 解析 JS、核对 id 与端点（13 项）
tools/blit_check.js          从 index.html 抠出 blit() 并在 node 里跑
tools/check_blit.py          真实帧走完 模拟器→HTTP→canvas（4 项）
tools/dashboard_selftest.py  进程内验证看板全部路由（18 项）
tools/check_deps.py          pyproject 声明与实际 import 双向一致（6 项）
tools/bench_brain.py         实测 laya 延迟 / input_tokens / answer_confidence
tools/run_headless.py        无服务无浏览器跑一段，按采样输出状态表 + 逐帧 PNG
tools/smb1/                  SMB1 专属标定：find_ram / verify_ram / dump_ram / probe_*
logs/decisions.jsonl         每次决策的完整输入输出
```

## 测试现状

五套测试都不需要商业 ROM，全部通过：

```
tools/smoke_test.py         20/20   模拟器、位表、按键往返、解码、观测契约、画像驱动的问法与规则、Flask 路由
tools/dashboard_selftest.py 18/18   全部路由、控制指令、线程存活、决策速率
tools/check_dashboard.py    13/13   JS 语法、DOM id、端点一致性、canvas 尺寸、RGB→RGBA
tools/check_blit.py          4/4    真实帧走完 模拟器→HTTP→canvas，逐字节等于标准 RGBA
tools/check_deps.py          6/6    pyproject 声明与实际 import 双向一致
tools/probe_emulator.py            cpu / ctrl / ppu / nmi 全 PASS
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

## 视觉层踩过的坑（都是静默的）

照常返回数字、照常渲染，只是返回的是错的。

**1. 地面剖面取错了对象。** 原来取「最低实心像素」。但地板延伸到屏幕底部时每一列在
239 行都是实心的，于是全屏地表=239。后果有两层，都很安静：角色站在 196 的地板上，
被量成「离地 44px」→ 永久判成腾空；40px 的前瞻窗口里没有任何列有地表 → 平坦地面
被判成坑。现在取「最低连续实心段的**顶**」，即真正落脚的表面。

**2. 地面色采到了天空。** 采样带写在表面上方两行——但表面正是落脚点，上面两行是
天空和站在上面的人。于是「地面色」= 天空色，减完之后整个 playfield 只剩一块
256px 宽的巨块，角色检测和物体检测一起死。改成采表面**下方**四行。

**3. 角色和它脚下的地板是连通的。** 不切掉地表以下的像素，8 连通域会把角色和地板
连成一块横跨全屏的 slab，而这一列的地表也被抬到角色头顶——于是角色自己也被后续
判据过滤掉。现在逐列按地表切。

**4. off-by-32 与天空写死。** 这两处在通用化之前就修过，动机相同：`SCREEN_H - 1`
应该换成裁剪后的行数，天空色应该从画面推断。见 `ground_profile()` 与
`background_colour()` 的注释。

---

## 推理延迟：一个真实的权衡

`tools/bench_brain.py` 在 CPU 上实测（multilingual 322M，3 个问题）：

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

置信度下限因此是 0.40 而不是 0.45：实测典型值就是 0.44，下限设 0.45 会把几乎
每一次决策都交给规则层，hybrid 等于退化成纯 reflex。

0.29 Hz 意味着**每 210 个 NES 帧才一次决策**。架构上靠「降频决策 + 保持上次按键」
撑住，这正是当初把两个线程解耦的原因。

---

## 已知限制

- **`score` 是 laya 自己文档里说的最弱原语。** `risk` 因此只用来辅助：它不直接驱动
  按键，只在 `move`/`jump` 都不确定时参与仲裁。
- **决策率上不去。** 322M 模型在 CPU 上就是几 Hz。这不是工程没做好，是物理限制，
  所以架构选择是「降频决策 + 保持输入」而不是「每帧决策」。
- **`generic` 画像分不出敌人种类，也分不出敌人和障碍物。** 视觉层给的是位置（精确）
  和高度，语义是空的。这是「不写死游戏知识」的代价。
- **角色定位只在横版过关上成立**（见上文）。垂直卷轴/俯视需要新的画像与判据。
- **可攀爬判定是形状启发式**，不是语义：一根又高又窄、游戏其实要你跳的柱子会被
  报成可攀爬。像素里分不出「意图」。
- **laya 的 checkpoint 首次运行要下 1.2–1.7 GB。** 这个步骤很可能需要重试或手动
  `hf download`。
