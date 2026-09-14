# DG-DRL

### 先把设计做完，再让仿真器开口

DG-DRL（Delayed-Gratification Deep Reinforcement Learning）是一套面向离散逆向设计的强化学习思路。

我们阅读了许多RL用于离散逆向设计的文献后发现一个问题：许多方法每调整一步结构就调用仿真，计算开销高耗时多，这种策略的样本效率极低。

DG-DRL 的思路是按照既定顺序逐个确定变量，每一步筛除非法动作，完整结构才调用一次仿真。其本质是“延迟满足”式奖励机制：把有限的成本集中到值得仿真的设计上，从而提高样本利用率。

听起来似乎朴素，但确实好用:)


## 三个机制



SDAS（Sequentially Decomposed Action Space）把高维离散设计拆成**有顺序**的变量决策。每一步只处理当前变量。当前变量确定后，再轮到下一个。所有变量都决定完，才得到完整设计。



DAM（Dynamic Action Masking）在动作选择前检查约束，把当前不可行的选项直接屏蔽掉。策略网络可以看见完整动作空间，但真正做选择时，只在合法动作里挑。少走弯路，也少浪费仿真。



DRM（Delayed Reward Mechanism）让前面的设计步骤保持0奖励。等完整结构组装完成后，返回最终性能作为延迟奖励。



## 代码在哪里？

~~~text
src/dgdrl/
├── core.py       # 顺序决策、动态约束和终端评估
└── masking.py    # 动作掩码与 masked selection 工具

examples/
└── method_demo.py

tests/
└── test_core.py

data/              # 少量项目参考数据
LICENSE            # MIT License
~~~

重点看 SequentialTerminalEnv。



## 跑个小例子

在仓库根目录执行：

~~~powershell
python -m pip install -r requirements.txt
python examples/method_demo.py
python -m pytest -q
~~~

这个 demo 使用的是一个玩具评估器。



## 接入自己的评估器

评估器就是一个普通的 Python callable。

~~~python
def evaluate_complete_design(design):
    # 换成你的电磁仿真、代理模型或实验接口
    return expensive_evaluation(design)
~~~

然后把它交给环境：

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



## 许可证

源代码采用 MIT License，详见 [LICENSE](LICENSE)。

外部软件和数据文件仍按各自适用的条款使用。

---

## English translation

### Finish the design before asking the simulator.

DG-DRL (Delayed-Gratification Deep Reinforcement Learning) is a reinforcement-learning idea for discrete inverse design.

After reading a range of reinforcement-learning approaches for discrete inverse design, we noticed a common problem: many methods call the simulator after every small structural update. That makes the workflow expensive, slow, and often wasteful in terms of samples.

DG-DRL decides the variables in a fixed order, filters invalid actions at each step, and calls the simulator only after the complete design has been assembled. In other words, it lets the limited simulation budget focus on designs that are worth evaluating.

It sounds simple, but it works. :)

## Three mechanisms

SDAS (Sequentially Decomposed Action Space) turns a high-dimensional discrete design into an ordered sequence of variable-wise decisions. Each step handles one variable; once all variables are selected, the complete design is ready.

DAM (Dynamic Action Masking) checks constraints before action selection and masks options that are invalid for the current state. The policy may see the full action space, but it only chooses from feasible actions.

DRM (Delayed Reward Mechanism) keeps the earlier design steps at zero reward. Once the complete structure is assembled, the evaluator is called and its final performance becomes the delayed reward.

## Where is the code?

The main entry point is `SequentialTerminalEnv` in src/dgdrl/core.py. The action-selection helpers are in src/dgdrl/masking.py. A small toy example is provided in examples/method_demo.py, with basic contract tests in tests/test_core.py.

## Quick start

From the repository root:

~~~powershell
python -m pip install -r requirements.txt
python examples/method_demo.py
python -m pytest -q
~~~

The demo uses a toy evaluator and does not require MATLAB or any external simulator.

## Use your own evaluator

The evaluator is simply a Python callable. It receives the complete design and returns one scalar score:

~~~python
def evaluate_complete_design(design):
    # Replace this with your electromagnetic simulator,
    # surrogate model, or experimental interface.
    return expensive_evaluation(design)
~~~

Pass it to the environment together with the action domains. If some actions violate fabrication or physical constraints, add a dynamic feasibility function. The environment handles the ordered decisions, action masking, and delayed terminal reward.

## License

The source code is released under the MIT License; see [LICENSE](LICENSE).

External software and data files remain subject to their own applicable terms.
