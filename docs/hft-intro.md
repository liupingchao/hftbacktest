# HftBacktest 项目结构与核心逻辑梳理

本文基于当前仓库代码梳理 HftBacktest 的模块结构、主要数据流和执行逻辑，帮助快速定位“数据如何进入系统、策略如何推进时间、订单如何模拟成交、Python 与 Rust 如何衔接、实时交易如何复用同一套策略接口”。

## 1. 项目定位

HftBacktest 是一个面向高频交易和做市策略的 tick 级回测框架。它的重点不是简单按 OHLCV 撮合，而是尽量复现真实交易中的几个关键因素：

- 行情从交易所产生到本地接收的 feed latency。
- 订单从本地下发到交易所、交易所处理、响应回到本地的 order latency。
- 限价单在盘口队列中的位置，以及盘口成交/撤单/数量变化对排队位置的影响。
- Level-2 Market-By-Price 与 Level-3 Market-By-Order 盘口重建。
- 多资产、多交易所统一推进。
- Rust 原生策略与 Python/Numba 策略两套使用路径。
- Rust live bot 使用与回测相近的 `Bot` 接口，便于从回测迁移到实盘原型。

## 2. 顶层结构

仓库是一个 Rust workspace，同时包含 Python 绑定、数据采集器、实盘连接器、文档和示例。

```text
.
├── Cargo.toml                    # Rust workspace 定义
├── README.rst                    # 项目总览与 Python 用户入口
├── hftbacktest/                  # 核心 Rust crate
├── hftbacktest-derive/           # Rust proc-macro，生成 NPY dtype 与资产构建分派代码
├── py-hftbacktest/               # Python 包与 PyO3/maturin 绑定
├── connector/                    # live connector，可连接 Binance Futures/Spot、Bybit
├── collector/                    # 行情采集 CLI，采集交易所 websocket/http 数据落盘
├── examples/                     # Python 示例、notebook、binance_tick_mm 示例工程
├── hftbacktest/examples/         # Rust 示例
└── docs/                         # Sphinx 文档、设计/计划文档与本文档
```

workspace 成员包括：

- `hftbacktest`: 核心库，提供回测、盘口、live bot、公共类型。
- `hftbacktest-derive`: 派生宏和构建宏。
- `py-hftbacktest`: Python 扩展模块和 Python 侧友好封装。
- `collector`: 行情采集工具。
- `connector`: 实盘连接器进程。

## 3. 核心 Rust crate：`hftbacktest`

`hftbacktest/src/lib.rs` 对外暴露主要模块：

```text
hftbacktest/src
├── backtest/       # 回测框架、资产构建、数据读取、撮合/成交模型
├── depth/          # 盘口数据结构和 L2/L3 盘口接口
├── live/           # live bot、IPC channel、实盘 recorder
├── types.rs        # Event、Order、LiveEvent、Bot trait 等公共类型
├── prelude.rs      # 常用类型 re-export
└── utils/          # 对齐数组等底层工具
```

### 3.1 公共类型：`types.rs`

`types.rs` 是跨回测、live、Python 绑定共用的基础协议层。

关键类型：

- `Event`: 行情事件，C layout、64 字节对齐，可直接映射 `.npy/.npz` 或 Python `numpy` 结构数组。
- `Order`: 订单状态，包含价格 tick、数量、剩余数量、成交数量、订单状态、请求状态、maker/taker 信息等。
- `LiveEvent`: connector 推给 live bot 的事件，包括行情、订单、仓位、错误。
- `LiveRequest`: live bot 发给 connector 的请求，包括注册品种、下单/撤单。
- `Bot<MD>` trait: 回测和 live bot 共享的策略接口。

`Event` 的重要字段：

```text
ev        # 事件类型 bit flags，例如 DEPTH_EVENT/TRADE_EVENT + LOCAL_EVENT/EXCH_EVENT + BUY/SELL
exch_ts   # 交易所事件时间
local_ts  # 本地接收时间
px        # 价格
qty       # 数量
order_id  # L3 MBO 事件使用
ival/fval # 预留扩展字段
```

`LOCAL_EVENT` 和 `EXCH_EVENT` 很关键：同一份历史行情会分别被“本地视角 processor”和“交易所视角 processor”按不同时间戳消费。

### 3.2 盘口模块：`depth/`

`depth/mod.rs` 定义统一盘口接口：

- `MarketDepth`: 查询 best bid/ask、tick size、lot size、指定 tick 数量。
- `L2MarketDepth`: 更新/清空 Market-By-Price 盘口。
- `L3MarketDepth`: 增删改 Market-By-Order 订单簿。
- `ApplySnapshot`: 应用初始快照并导出当前快照。

主要实现：

- `HashMapMarketDepth`: 基于 hash map 的通用 L2 盘口。
- `ROIVectorMarketDepth`: 只维护 range of interest 范围内价格层，适合高性能回测。
- `BTreeMarketDepth`: 基于 BTree 的盘口。
- `FusedHashMapMarketDepth`: live connector 侧用于融合不同深度/频率行情流。

### 3.3 回测模块：`backtest/`

`backtest/` 是项目核心。

```text
hftbacktest/src/backtest
├── mod.rs                 # Asset builder、Backtest、主事件循环
├── data/                  # npy/npz 读取、Data、Reader、预处理器
├── proc/                  # Local/Exchange processor
├── models/                # latency、queue、fee 模型
├── order.rs               # Local <-> Exchange 订单延迟总线
├── state.rs               # 仓位、余额、手续费、交易量等状态
├── assettype.rs           # Linear/Inverse 合约权益计算
├── recorder.rs            # 回测统计记录
└── evs.rs                 # 多资产事件优先级集合
```

#### 3.3.1 Asset 构建

回测中每个资产不是一个单一对象，而是由三部分组成：

- `local`: 本地视角 processor，维护策略看到的盘口、订单状态、仓位。
- `exch`: 交易所视角 processor，模拟订单在交易所的接收、排队和成交。
- `reader`: 行情数据 reader，按批读取 `Event` 数据。

`Asset::l2_builder()` 和 `Asset::l3_builder()` 负责组装：

- 数据源：文件或内存数据。
- 资产类型：`LinearAsset` / `InverseAsset`。
- 盘口结构：`HashMapMarketDepth` / `ROIVectorMarketDepth` 等。
- 延迟模型：`ConstantLatency` / `IntpOrderLatency`。
- 队列模型：`RiskAdverseQueueModel` / `ProbQueueModel` / `L3FIFOQueueModel` 等。
- 交易所模型：`NoPartialFillExchange` / `PartialFillExchange` / L3 exchange。
- 手续费模型：按交易价值、交易数量或固定每笔。

构建时会创建双向订单总线：

```text
LocalToExch: 本地请求 -> 交易所接收
ExchToLocal: 交易所响应 -> 本地接收
```

订单总线用 latency model 计算到达时间，因此订单请求和响应会作为时间轴上的事件参与调度。

#### 3.3.2 回测主循环

`Backtest<MD>` 内部为每个资产维护两条 processor state：

```text
local[asset_no]  # 本地视角
exch[asset_no]   # 交易所视角
```

`EventSet` 为每个资产维护 4 类下一事件时间：

```text
LocalData   # 本地接收行情事件，按 event.local_ts
LocalOrder  # 本地收到订单响应
ExchData    # 交易所发生行情事件，按 event.exch_ts
ExchOrder   # 交易所收到订单请求
```

每次推进时，`EventSet::next()` 找出全局最早事件，然后按类型处理：

1. `LocalData`
   - 本地 processor 消费行情。
   - 更新策略可见盘口、last trades、feed latency。
   - 读取下一条本地可见行情。

2. `ExchData`
   - 交易所 processor 消费行情。
   - 更新交易所侧盘口。
   - 根据盘口变动和成交事件判断挂单是否成交。
   - 如果有成交/订单状态变化，写入 exchange-to-local 订单总线。

3. `ExchOrder`
   - 交易所收到本地订单请求。
   - 根据当前交易所盘口、订单类型、TIF、队列模型决定接受、过期、成交或部分成交。
   - 生成响应并写入 exchange-to-local 总线。

4. `LocalOrder`
   - 本地收到交易所订单响应。
   - 更新本地订单状态。
   - 如果成交，更新本地 `State` 中的仓位、余额、手续费、交易量。

策略通常通过以下方法推进时间：

- `elapse(duration)`: 推进指定时间。
- `wait_next_feed(include_order_resp, timeout)`: 等到下一条行情或超时。
- `wait_order_response(asset_no, order_id, timeout)`: 等指定订单响应。
- `goto_end()`: 处理到数据结束。

这也是 HFT 回测区别于普通 bar 回测的核心：时间不是单纯由 K 线推进，而是由行情事件、订单请求到达、订单响应到达共同驱动。

#### 3.3.3 Local processor

`proc/local.rs` 中的 `Local` 表示策略本地看到的世界：

- 只处理带 `LOCAL_EVENT` 的行情，时间使用 `local_ts`。
- 维护策略可见盘口。
- 保存最近交易事件。
- 管理本地订单表。
- 发送下单、改单、撤单请求到 `LocalToExch`。
- 接收交易所响应后更新本地订单状态。
- 成交响应到达本地时更新账户状态。

本地侧下单不会立即改变交易所状态，而是先通过 order latency 排入订单总线。

#### 3.3.4 Exchange processor

`proc/nopartialfillexchange.rs` 和 `proc/partialfillexchange.rs` 表示交易所模拟侧：

- 只处理带 `EXCH_EVENT` 的行情，时间使用 `exch_ts`。
- 维护交易所侧盘口。
- 收到订单请求后按当前盘口判断：
  - 是否吃单成交。
  - post-only 是否过期。
  - 是否进入挂单队列。
  - FOK/IOC/GTC/GTX 在当前模型下如何处理。
- 行情成交和盘口数量变化会驱动 queue model 更新订单排队位置。
- 成交或状态变化后，通过 `ExchToLocal` 按 response latency 发送响应。

`NoPartialFillExchange` 与 `PartialFillExchange` 的差异：

- `NoPartialFillExchange`: 满足成交条件时直接全量成交。
- `PartialFillExchange`: 同价位队列前方数量被打穿时，可以按剩余成交量部分成交。

L3 路径由 `l3_local.rs` 和 `l3_nopartialfillexchange.rs` 提供，依赖 Market-By-Order 事件和 `L3FIFOQueueModel`。

#### 3.3.5 Latency / Queue / Fee / State

`models/latency.rs`：

- `ConstantLatency`: 固定 entry latency 与 response latency。
- `IntpOrderLatency`: 从历史订单延迟数据插值，支持 latency offset。
- 负 entry latency 表示订单在到达撮合前因技术原因被拒绝，本地在 `local_ts - latency` 收到拒绝。

`models/queue.rs`：

- `RiskAdverseQueueModel`: 只有同价成交才推进队列位置，偏保守。
- `ProbQueueModel`: 同价成交和盘口数量减少都会影响队列位置，具体概率函数可替换。
- `LogProbQueueFunc*`、`PowerProbQueueFunc*`: 不同队列概率函数。
- `L3FIFOQueueModel`: L3 MBO 下直接利用订单级队列。

`models/fee.rs`：

- `TradingValueFeeModel`
- `TradingQtyFeeModel`
- `FlatPerTradeFeeModel`

`state.rs`：

- `State` 记录 `position`、`balance`、`fee`、`num_trades`、`trading_volume`、`trading_value`。
- `apply_fill()` 在成交时按资产类型和手续费模型更新账户状态。

### 3.4 数据读取：`backtest/data/`

数据层围绕 `Data<Event>` 和 `Reader<Event>` 实现。

- `Data` 是对底层连续内存的轻量视图，支持从文件或 Python ndarray 映射。
- `DataPtr` 使用 cache-line 对齐内存，提高顺序访问效率。
- `NpyDTyped` 描述 Rust struct 对应的 NumPy dtype。
- `read_npy_file` / `read_npz_file` 读取 `.npy/.npz`。
- `Reader` 管理多文件顺序读取、可选 parallel load、数据 release。
- `FeedLatencyAdjustment` 可对行情 local timestamp 做 offset，用于跨机房/跨交易所校准。

## 4. Python 包：`py-hftbacktest`

Python 包由两层构成：

```text
py-hftbacktest
├── Cargo.toml / src/       # PyO3 扩展模块 hftbacktest._hftbacktest
└── hftbacktest/            # Python 封装、Numba jitclass、数据工具、统计工具
```

### 4.1 PyO3 扩展层

`py-hftbacktest/src/lib.rs` 暴露 Python 可见类和函数：

- `BacktestAsset`: Python 侧资产配置对象。
- `build_hashmap_backtest`
- `build_roivec_backtest`
- live feature 开启时的 `build_hashmap_livebot` / `build_roivec_livebot`

`BacktestAsset` 在 Python 中以链式 API 设置数据、资产类型、延迟模型、队列模型、交易所模型、tick/lot size、ROI 范围等。最终 build 函数会调用 Rust `Asset` builder 创建 `Backtest<MD>`，并把裸指针交给 Python binding 层。

`hftbacktest-derive::build_asset!` 用宏生成多种 asset type、latency model、queue model、exchange model、fee model 组合的 Rust match 分派，避免手写大量重复构建代码。

### 4.2 Python/Numba binding 层

`py-hftbacktest/hftbacktest/binding.py` 使用 `ctypes.CDLL` 加载 `_hftbacktest` 动态库中的 `extern "C"` 函数，并包装成 Numba `jitclass` 可调用对象。

调用链大致是：

```text
Python BacktestAsset
  -> PyO3 BacktestAsset
  -> build_hashmap_backtest / build_roivec_backtest
  -> Box<Backtest<MD>> raw pointer
  -> binding.py jitclass 保存 ptr
  -> Numba 策略中调用 hbt.elapse(), hbt.depth(), hbt.orders(), hbt.submit_*()
  -> C ABI 函数
  -> Rust Backtest<MD>
```

这种设计的目的：

- 策略主体可以写在 Python + Numba `@njit` 中。
- 高性能状态推进、盘口、订单模拟仍由 Rust 执行。
- Python 侧保留 NumPy 数据、统计、绘图、notebook 工作流。

### 4.3 Python 工具目录

`py-hftbacktest/hftbacktest/` 包含：

- `data/utils/`: 不同数据源转换工具，例如 Binance Futures、Bybit、Tardis、Databento、Hyperliquid、MEXC 等。
- `data/validation.py`: 数据校验。
- `stats/`: 回测统计指标与分析。
- `recorder.py`: Python 侧记录器。
- `types.py` / `order.py` / `state.py`: Python 常量、dtype、结构映射。

## 5. Live bot 与 connector

实时交易分成两部分：

- `hftbacktest/src/live/`: bot 侧统一接口。
- `connector/`: 独立连接器进程，负责交易所 API、websocket、REST、订单管理、IPC 发布。

### 5.1 LiveBot

`LiveBot<CH, MD>` 实现与回测相同的 `Bot<MD>` trait，因此策略代码在 Rust live 和 Rust backtest 之间可以复用大量接口：

- `depth(asset_no)`
- `orders(asset_no)`
- `position(asset_no)`
- `submit_buy_order` / `submit_sell_order`
- `cancel`
- `wait_next_feed`
- `wait_order_response`
- `elapse`

live 模式下，时间由真实时间和 connector 推送事件驱动。`Instrument` 保存每个交易标的的：

- connector 名称。
- symbol。
- tick size / lot size。
- 当前盘口。
- 本地订单表。
- 最新 feed/order latency。
- 仓位状态。

### 5.2 IPC channel

live bot 与 connector 使用 `iceoryx2` IPC 通信：

```text
LiveBot -- LiveRequest --> connector
LiveBot <-- LiveEvent ---- connector
```

注册品种时，connector 会向 bot 发送一个 batch：

1. `BatchStart`
2. 当前 working orders
3. 当前 position
4. 当前 market depth snapshot
5. `BatchEnd`

这样新启动的 bot 可以先同步状态，再开始处理增量事件。

### 5.3 connector

`connector/src/main.rs` 是连接器 CLI 主入口。它完成：

- 读取 connector 配置。
- 根据 feature/配置创建 Binance Futures、Binance Spot 或 Bybit connector。
- 接收 bot 发来的注册、下单、撤单请求。
- 调用交易所 REST/websocket API。
- 将行情、订单回报、仓位和错误封装成 `LiveEvent` 发布给 bot。
- 维护 `FusedHashMapMarketDepth`，可融合不同深度/频率的行情流，并向新 bot 提供快照。

`connector/src/connector.rs` 定义统一接口：

- `ConnectorBuilder`
- `Connector`
- `GetOrders`
- `PublishEvent`

各交易所目录：

```text
connector/src
├── binancefutures/
├── binancespot/
└── bybit/
```

每个交易所通常拆成：

- REST 客户端。
- public market data stream。
- private/user data stream。
- order manager。
- 消息结构。

## 6. collector：行情采集工具

`collector/` 是独立 CLI，用于采集原始交易所数据。

入口：`collector/src/main.rs`

命令参数：

```text
collector <path> <exchange> <symbols...>
```

支持的 exchange 分支包括：

- `binancefutures` / `binancefuturesum`
- `binancefuturescm`
- `binance` / `binancespot`
- `bybit`
- `hyperliquid`

采集流程：

1. 根据交易所选择 websocket topics/streams。
2. 启动对应交易所的 collection task。
3. 收到数据后通过 channel 发给 writer。
4. `file::Writer` 按路径和 symbol 落盘。
5. 用户后续可用 Python data utils 转换为 HftBacktest `Event` 格式。

collector 更偏“原始数据采集”，而 `py-hftbacktest/hftbacktest/data/utils/` 更偏“转换为回测标准事件格式”。

## 7. hftbacktest-derive

`hftbacktest-derive/` 提供两个主要宏：

- `#[derive(NpyDTyped)]`: 为 Rust C-layout struct 生成 NumPy dtype 描述，供 `.npy/.npz` 读写和 Python 映射使用。
- `build_asset!`: 在 Python binding 构建资产时，生成多种模型组合的分派代码。

这个 crate 主要服务于“Rust 内存结构与 Python/NumPy 结构数组一致”以及“Python 配置对象映射到 Rust 泛型 builder”的问题。

## 8. 一次典型 Python 回测流程

以 Python 用户视角，流程通常是：

1. 用数据工具把交易所原始行情转成 `Event` `.npz` 或 NumPy structured array。
2. 创建 `BacktestAsset()`，配置：
   - `data(...)`
   - `linear_asset(...)` 或 `inverse_asset(...)`
   - `constant_latency(...)` 或 `intp_order_latency(...)`
   - queue model
   - exchange model
   - fee model
   - `tick_size` / `lot_size`
3. 用 `HashMapMarketDepthBacktest([asset])` 或 `ROIVectorMarketDepthBacktest([asset])` build 回测器。
4. 在 `@njit` 策略函数中循环调用：
   - `hbt.elapse(...)` 或 `hbt.wait_next_feed(...)`
   - `hbt.depth(asset_no)` 读取盘口
   - `hbt.orders(asset_no)` 管理挂单
   - `hbt.submit_*` / `hbt.cancel`
5. 使用 `Recorder` 或 stats 模块统计结果。

内部实际执行链：

```text
Numba strategy
  -> binding.py jitclass
  -> C ABI
  -> Rust Backtest
  -> EventSet 按时间推进
  -> Local/Exchange processor 消费行情和订单事件
  -> State/Order/Depth 更新
```

## 9. 一次典型 Rust live 流程

Rust live 原型通常分两端：

1. 启动 connector 进程：
   - 读取交易所 API 配置。
   - 建立 public/private stream。
   - 对外提供 IPC channel。

2. 策略进程创建 `LiveBot`：
   - 注册 `Instrument`。
   - 通过 IPC 接收初始 batch。
   - 循环 `wait_next_feed` 或 `elapse`。
   - 按相同 `Bot` trait 下单和撤单。

核心思想是：回测和 live 都面向 `Bot<MD>` 接口，只是事件来源不同。

```text
Backtest: Reader<Event> + latency/order simulation
Live:     Connector IPC + real exchange responses
```

## 10. examples 与 docs

主要示例分布：

- `examples/*.ipynb`: Python notebook 教程，如做市、Level-3、Pricing Framework、多市场等。
- `examples/binance_tick_mm/`: 一个较完整的 Binance tick-level market making 示例工程，包含回测、live、audit、walk-forward、sweep、对齐分析等脚本。
- `hftbacktest/examples/*.rs`: Rust 回测和 live 示例。
- `connector/examples/*.toml`: connector 配置样例。
- `docs/`: Sphinx 文档源，包括 data、latency models、order fill、market maker program、reference 等。

当前仓库还有一些 `docs/superpowers/` 和 `TODO_live_open_order_alignment.md`，看起来是近期围绕 live/backtest 对齐、audit cadence、partial fill 对齐等工作的设计和计划文档。

## 11. 重要扩展点

如果要扩展框架，常见位置如下：

- 新数据源转换：
  - Python: `py-hftbacktest/hftbacktest/data/utils/`
  - 原始采集: `collector/src/<exchange>/`

- 新交易所 live connector：
  - 新建 `connector/src/<exchange>/`
  - 实现 `ConnectorBuilder` 和 `Connector`
  - 实现订单管理、public/private stream、REST 签名。

- 新盘口结构：
  - 实现 `MarketDepth`，按需要实现 `L2MarketDepth` / `L3MarketDepth` / `ApplySnapshot`。

- 新订单延迟模型：
  - 实现 `LatencyModel`，返回 entry/response latency。

- 新队列模型：
  - L2 实现 `QueueModel<MD>`。
  - L3 实现对应 L3 queue trait。

- 新手续费模型：
  - 实现 `FeeModel`。

- 新资产类型：
  - 实现 `AssetType`，定义 amount/equity 计算。

- Python 侧暴露新的 Rust 模型：
  - 更新 PyO3 enum/config。
  - 更新 `build_asset!` 可选组合。
  - 在 Python `__init__.py` 或 binding 层补充 API。

## 12. 需要注意的设计取舍

- 回测假设市场回放本身不受你的订单影响。大额吃单或明显影响盘口的策略会更容易失真。
- `OrderBus` 为简化回测，保证订单请求/响应时间不倒序；真实 REST API 下可能存在后发先至。
- `NoPartialFillExchange` 会在满足条件时全量成交，适合保守或简化场景；更接近真实部分成交应使用 `PartialFillExchange`，但仍需用 live 结果校准。
- feed latency 和 order latency 的时间单位必须与数据时间戳一致，项目文档和代码倾向使用纳秒。
- Python binding 通过裸指针和 C ABI 连接 Rust 对象，性能高，但生命周期、dtype 对齐和内存布局必须保持一致。
- live bot 与 backtest 共享接口，但 live 依赖真实交易所回报，订单状态可能出现乱序、延迟、拒单、连接中断等，需要 error handler 和 audit 机制配合。

## 13. 快速阅读建议

如果要继续深入代码，建议按以下顺序读：

1. `hftbacktest/src/types.rs`: 先理解 `Event`、`Order`、`Bot`。
2. `hftbacktest/src/backtest/mod.rs`: 理解 `Asset` builder 和 `Backtest::goto` 主循环。
3. `hftbacktest/src/backtest/proc/local.rs`: 理解策略本地视角。
4. `hftbacktest/src/backtest/proc/nopartialfillexchange.rs` 与 `partialfillexchange.rs`: 理解成交模拟。
5. `hftbacktest/src/backtest/order.rs`: 理解订单延迟总线。
6. `hftbacktest/src/backtest/models/latency.rs` 与 `queue.rs`: 理解延迟和排队模型。
7. `py-hftbacktest/src/lib.rs` 与 `py-hftbacktest/hftbacktest/binding.py`: 理解 Python 如何调用 Rust。
8. `hftbacktest/src/live/bot.rs` 与 `connector/src/main.rs`: 理解实盘事件流。

一句话总结：HftBacktest 的核心抽象是“同一份行情同时驱动本地视角和交易所视角，订单通过带延迟的双向总线在两者之间流动，策略只面对统一的 `Bot` 接口”。这个结构让它能在 tick 级回放中显式模拟 feed latency、order latency、队列位置和订单响应时序。
