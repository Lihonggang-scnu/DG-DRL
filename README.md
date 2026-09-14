# DG-DRL

### 先把设计做完，再让仿真器开口
### Finish the design before asking the simulator.

DG-DRL（Delayed-Gratification Deep Reinforcement Learning）是一套面向离散逆向设计的强化学习思路。

说白了：别每改一点结构就跑一次昂贵仿真。先把整套设计按顺序拼好，再统一问一句——“这版到底行不行？”

听起来似乎朴素，但确实好用。:)

DG-DRL is a reinforcement-learning framework for discrete inverse design.

The basic idea is simple: do not run an expensive simulation after every tiny
structural change. Build the complete design first, then ask the simulator once
whether the whole thing works.

It may sound modest, but it works surprisingly well. :)

## 核心思想 / Core idea

传统流程 / Traditional workflow:

~~~text
改一点 → 仿真一次 → 得到即时奖励
Change a little → simulate → receive an immediate reward
~~~

DG-DRL 流程 / DG-DRL workflow:

~~~text
逐个决定设计变量
Decide the variables one by one
          ↓
筛掉当前不合规的动作
Mask invalid actions
          ↓
得到完整结构
Build the complete design
          ↓
只做一次昂贵评估
Run one expensive evaluation
          ↓
返回终端奖励
Return the terminal reward
~~~

## 三个机制 / Three mechanisms

### SDAS：按变量，一个一个来

SDAS（Sequentially Decomposed Action Space）把高维离散设计拆成有顺序的变量决策。每一步只处理当前变量，所有变量都决定完后才得到完整设计。

SDAS decomposes a high-dimensional discrete design into ordered variable-wise
decisions. Each step handles one variable, and the complete design appears at
the end of the sequence.

### DAM：不合规的动作，先请出去

DAM（Dynamic Action Masking）在动作选择前检查约束，把当前不可行的选项直接屏蔽掉。策略只在合法动作里挑，少走弯路，也少浪费仿真。

DAM checks constraints before action selection and masks options that are not
currently feasible. The policy chooses only from the valid actions.

### DRM：奖励晚点到，但一次说清楚

DRM（Delayed Reward Mechanism）让前面的步骤先不急着领奖励。等完整结构拼好后，评估器只调用一次，返回最终性能作为终端奖励。

DRM keeps intermediate rewards at zero. Once the full design is assembled, the
evaluator is called once and its final score becomes the terminal reward.

## 代码在哪里？ / Where is the code?

~~~text
src/dgdrl/
├── core.py       # 顺序决策、动态约束和终端评估
└── masking.py    # 动作掩码与 masked selection 工具

examples/
└── method_demo.py

tests/
└── test_core.py

data/              # 少量项目参考数据 / Small project reference data
LICENSE            # MIT License
~~~

重点看 SequentialTerminalEnv。它不绑定具体物理求解器，也不强行规定强化学习算法。你只需要提供动作域、可行性规则和一个完整设计评估器。

The main entry point is SequentialTerminalEnv. It is solver-agnostic and does
not require a particular reinforcement-learning algorithm. Provide the action
domains, feasibility rules, and an evaluator for a complete design.

## 跑个小例子 / Quick start

在仓库根目录执行 / From the repository root:

~~~powershell
python -m pip install -r requirements.txt
python examples/method_demo.py
python -m pytest -q
~~~

这个 demo 用的是玩具评估器，不需要 MATLAB，放心跑。

The demo uses a toy evaluator. No MATLAB or external simulator is needed.

## 接入自己的评估器 / Bring your own evaluator

评估器就是一个普通的 Python callable。它接收完整设计，返回一个标量。

An evaluator is just a Python callable. It receives a complete design and
returns one scalar score.

~~~python
def evaluate_complete_design(design):
    # 换成你的电磁仿真、代理模型或实验接口
    # Replace this with your simulator, surrogate, or experiment
    return expensive_evaluation(design)
~~~

然后把它交给环境 / Then pass it to the environment:

~~~python
from dgdrl import SequentialTerminalEnv

env = SequentialTerminalEnv(
    action_domains=(
        [0, 1],
        [0, 1, 2],
        [0, 1],
    ),
    evaluator=evaluate_complete_design,
)
~~~

如果某些动作会违反制造或物理约束，再提供一个动态可行性函数即可。其余的顺序决策、动作掩码和终端奖励逻辑，环境会处理好。

If some actions violate fabrication or physical constraints, add a dynamic
feasibility function. The environment handles the ordered decisions, masking,
and terminal reward logic.

## 和论文的关系 / Relation to the paper

论文中，DG-DRL 被用于结构色薄膜、近场光束整形和自由形状光栅光束偏转等离散光子逆向设计任务。

In the paper, DG-DRL is applied to discrete photonic inverse-design tasks
including structural-color films, near-field beam shaping, and freeform-grating
beam steering.

这里公开的是方法骨架和交互接口，方便读者快速理解这个想法。论文实验中的具体参数、私有求解器配置、训练脚本、结果文件和模型 checkpoint 不在本仓库中。

This repository releases the method skeleton and interaction interface. The
paper-specific parameters, private solver configurations, training scripts,
results, and model checkpoints are intentionally not included.

## 许可证 / License

源代码采用 MIT License，详见 [LICENSE](LICENSE)。

The source code is released under the MIT License; see [LICENSE](LICENSE).

外部软件和数据文件仍按各自适用的条款使用。

External software and data files remain subject to their own applicable terms.
