# hub8002：谁是球队真正的核心？——控球网络推断与贝叶斯反事实价值

HKU DATA8002 小组项目课题 C 的代码仓库。数据为 StatsBomb 开放数据：巴萨 17 个西甲赛季（524 场）加 2015/16 西甲全部 380 场，去重后 866 场。展示或发表时须注明数据来源 StatsBomb，并放其 logo。本仓库不包含原始数据和由其生成的中间表，运行 `make data` 会从 StatsBomb 的公开仓库下载。

**最终方案（2026-11-08）**：核心 = 换成同位置普通球员时球队损失的预期进球（价值增量，VOA），分四类：进攻、推进、防守（节点层，Gamma–泊松贝叶斯模型）和串联（网络层，控球马尔可夫链上的反事实）。两层共用"同位置普通球员"零模型和同一条判定规则（核心集合 / 无明确核心）。主口径：平均水平基准、三类位置；替补水平基准为第二口径；四类、八类位置为稳健性分析。v1 的传球网络枢纽（Kirkley MDL 等）保留作对照。结果总结见 `results/SUMMARY_v2.md`、`SUMMARY_net.md`、`SUMMARY_v3.md`，数据核对见 `results/data_review.md`。

## 环境

本机的做法：在已有的 conda 环境（已装 numpy、scipy、pandas、networkx、matplotlib）上建 venv，再补装 pyarrow 和 pytest：

```bash
python -m venv --system-site-packages .venv && .venv/bin/pip install pyarrow pytest
echo "$PWD/src" > .venv/lib/python3.11/site-packages/hub8002_src.pth
```

组员从零开始可以用 `conda env create -f environment.yml`。用 matplotlib-base 而不是 matplotlib，避免拉进 Qt。

Kirkley 的官方 MDL 代码没有许可证文件，本仓库不附带；运行 `scripts/fetch_third_party.sh` 克隆到 `third_party/kirkley_hubs/`（v1 对照和 `tests/test_mdl.py` 需要）。

## 一键运行

```bash
make data    # 下载（约 2.6 GB）→ 5 张中间表 → 数据检查
make calib   # 联盟基准速率、过度离散 φ、个体差异 α
make sim     # E1、E2（16 进程约 30 分钟）
make real    # E3–E7（几分钟）
make figs    # figures/*.pdf|png 和 results/tables.md
make v2      # 价值核心（进攻 / 推进 / 防守）：值表、校准、真实数据、模拟、检验、图
make net     # 网络扩展（串联核心）：控球链、校准、真实数据、模拟、检验、图
make v3      # 改进与外部检验：替补水平、位置划分选择、缺阵检验、公认评价对照、数据核对、报告用图
make all     # 以上全部（12 进程约 1 小时）
make test    # 单元测试
```

`make clean-results` 清空 `results/`（保留手写的 `SUMMARY*.md`）和 `figures/`。

读取下载文件的 Python 一律用 `-I`（隔离模式）运行。每个进程设置 `OMP_NUM_THREADS=1`，以免和多进程抢占 CPU。

## 目录

| 路径 | 内容 |
|---|---|
| `config.yaml` | 开工前定的 12 个决定（位置映射、两层定义、先验、门槛等） |
| `src/hub8002/io_statsbomb.py` | 读取比赛列表、事件、阵容 |
| `src/hub8002/minutes.py` | 上场区间与分钟：按真实半场长度计时，红牌截断，修正"伤停离场后被换下却仍记为回场"的数据错误 |
| `src/hub8002/passes.py` | 传球表与两层标记（`is_org`、`is_box`，以及试跑定义 `is_box_pilot`） |
| `src/hub8002/aggregate.py` | 任意一组比赛 → 球员计数表和有向边表（含两人共同在场分钟） |
| `src/hub8002/models/bayes_hub.py` | `fit_hubs`（原方案的泊松–Gamma 模型）、`fit_hubs_het`（加入个体差异的负二项版本）、`log_ml_het_set`、`log_pred_individual` |
| `src/hub8002/models/mdl.py` | Kirkley MDL：官方 `hubs()` 加试跑移植版（100 个随机度序列上结果完全一致） |
| `src/hub8002/models/baselines.py` | 传球数第一、每 90 分钟第一、PageRank、均值 + 2SD、Average、Loubar、MDL 四种编码 |
| `src/hub8002/sim/` | 按真实 2015/16 名单校准的模拟器 |
| `src/hub8002/experiments/` | E1–E7 |
| `scripts/` | 下载、建表、检查、校准、画图 |
| `data/interim/` | matches、appearances、passes、match_edges、player_period、edges（parquet） |
| `results/` | 每个实验一个 json；`data_check.md`、`tables.md` |
| `figures/` | 图 1、3–8（pdf 和 png） |
| `third_party/kirkley_hubs/` | Kirkley 官方代码（不随仓库分发，用 `scripts/fetch_third_party.sh` 获取） |
| `docs/` | MDL 等价推导、公认评价参照（`consensus.yaml`）、换人时间外部对照（`minutes_external_check.yaml`） |

## v2（2026-11-07 修订，论文主线）：核心 = 对球队最有价值的人

核心分进攻、组织、防守三类。某一方面的核心，是全队中在这方面"超出联盟同位置平均水平的价值"（VOA，单位为预期进球）最多、且有把握为正的人；排名不确定时，并列的几人一起算作核心。

- 价值：`src/hub8002/value.py`。进攻 = 非点球 xG + xA；组织 = 后场和中场发起的传球与带球带来的 xT 增量（xT 用 866 场自行拟合）；防守 = 抢断、拦截、夺回球权、封堵、解围时消除的对手威胁。机会时间：进攻和组织用本队控球分钟，防守用对手控球分钟。
- 模型：`src/hub8002/models/core_model.py`（Gamma–泊松共轭，经验贝叶斯 α，后验抽样得到 P(#1) 和核心集合）。
- 实验：`experiments/v2_real.py`（巴萨、西甲）、`v2_sim.py`（模拟检验）、`v2_checks.py`（前后半季、逐场确认、敏感性）。
- 运行 `make v2`。结果见 `results/SUMMARY_v2.md`、`results/tables_v2.md`、`figures/v2_*.png`。

### 网络扩展（2026-11-08）：串联核心

在控球网络上计算价值：每次控球的持球人序列构成马尔可夫链，把球员"不射门时如何处理球"换成联盟同位置的平均行为，全队预期进球的减少量就是他的网络价值增量。它能看到 xT 看不到的"不丢球、传对人"。

- 控球链：`src/hub8002/chains.py`，`scripts/build_chains.py` → `data/interim/chains.parquet`。
- 模型：`src/hub8002/models/net_model.py`（Sherman–Morrison 闭式反事实，Dirichlet–多项分布经验贝叶斯，拟多项 φ，射门 xG 的不确定性，"不丢球 / 传给谁"拆分）。
- 校准：`scripts/calibrate_net.py` → `results/net_calib.json`。
- 实验：`experiments/net_real.py`、`net_sim.py`、`net_checks.py`（前后半季、敏感性）。
- 运行 `make net`。结果见 `results/SUMMARY_net.md`、`results/tables_net.md`、`figures/net_*.png`。

核心集合规则已于 2026-11-08 修订（v2 和网络版相同）：集合中只要有一人 P(VOA > 0) < 0.9，就报"无明确核心"。

### 改进与外部检验（2026-11-08，`make v3`）

主结果不变（平均水平基准、三类位置）；以下是新增的变体与检验，结果见 `results/SUMMARY_v3.md`、`results/tables_v3.md`。

- 替补水平基准：`scripts/calibrate_bench.py` 用 2015/16 首发不到一半的球员实测替补水平 → `results/bench_rates.json`；模型参数 `value_cores(voa_base=…)`、`net_cores(q_base=…)`（先验不变，只换比较对象）。模拟 `v2_sim … bench50`、`net_sim … bench50`；真实数据 `experiments/rep_real.py`。
- 位置划分由数据选择：`scripts/build_position_minutes.py`（按 StatsBomb 原始位置统计分钟）→ `experiments/partitions.py`（7 种划分，描述长度 + 留一队交叉验证）→ `experiments/partition_real.py`（按八类、七类、四类位置重算）。网络模型支持"比较对象"与"零模型接球分块"分开（`NetUnit.dst`）。
- 缺阵检验：`experiments/absence.py`（2015/16，控制对手与主客场，前后半季交叉拟合，队内置换检验）。
- 公认评价对照：`docs/consensus.yaml`（金球奖、LFP 奖项，带出处）+ `experiments/consensus.py`（另含各队射手王）。

下文的"两个模型"和 E1–E7 是 v1（传球网络枢纽），保留作对照。

## 两个模型（v1：传球网络枢纽）

原方案的模型是 `k_i ~ Poisson(T_i λ_g ρ^{s_i})`，见 `fit_hubs`。数据显示，同队同位置组的非核心球员之间存在稳定的风格差异：组织层赛季总数的离散度是泊松预期的 25–40 倍，远大于逐场过度离散 φ ≈ 3.2。原模型因此几乎每队都报出 3 个核心。

本模型 `fit_hubs_het` 给每名球员一个风格倍数 `u_i ~ Gamma(α_g, α_g)`，`α_g` 用 2015/16 全联盟按位置组做经验贝叶斯估计。核心的定义随之变为"明显超出同位置正常差异范围的人"。逐场过度离散仍按拟泊松方法处理，即用 `(k/φ, T/φ)` 代入。
