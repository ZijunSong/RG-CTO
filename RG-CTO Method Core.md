Main Experiments
三种方法都建立在 CTO 迭代流水线之上：先用 iter0 rollout 做经验蒸馏（提炼 verified propositions / critical pitfalls），再在后续迭代用 正/负对比 prompt 引导生成。区别在于：怎么采样轨迹、怎么高效打负分、什么时候该抑制负证据。
CTO-Rescore++（聚类代表负分重排）
核心思路：固定生成 N 条轨迹，但只对少量 聚类代表 做负向打分，降低计算量。
1. 正样本生成：用 positive prompt 生成 N 条完整推理轨迹  
2. 按答案聚类：提取最终答案，用 math_equal 将轨迹按等价答案分组  
3. 选代表：每个聚类选一个代表（如最高正分 / 最短可解析 / 语义质心）  
4. 负分打分：仅对 Top-K 个聚类的代表，用 negative prompt 做 teacher-forced 打分  
5. 分数传播：代表轨迹的负分复制给同聚类所有成员  
6. 重排：S(y) = logp_pos(y) − α·logp_neg(y_rep) + β·log|C(a)| + γ·Q_pos(y)
特点：在 Rescore 基础上用聚类压缩负分计算；通过 β·log|C(a)| 奖励多数派答案簇，提升稳定性。
CTO-Wave（渐进式波次生成）
核心思路：不一次性生成 N 条，而是 分波 rollout + 早停，必要时才触发负分重排。
1. 分波生成：按配置（如 8,8,16）逐波追加正样本轨迹  
2. 置信早停：每波后按答案聚类，若 多数比例 ≥ 阈值 且 top1−top2 差距 ≥ margin，提前停止  
3. 条件触发 Rescore++：当 top-2 簇接近（或配置允许时），对当前候选做一次 聚类代表负分（复用 Rescore++ 逻辑）  
4. 负风险早停（可选）：负分后若多数已高且负风险低，也可停止  
5. 最终重排：与 Rescore++ 相同的对比打分公式
特点：算力随难度自适应——简单题少采样，难题多采样；只在“答案不确定”时才做昂贵的负分。
TR-CTO（Trust-Region CTO，信任域对比解码）
核心思路：保留 CTO 的对比解码框架，但用 信任门控 动态调节负向抑制强度；负证据不可靠时自动退化为正样本主导。
对比分数：s = logp_pos − α_j · logp_neg
其中：α_j = α₀ · Trust(B⁻, q) · Locality(B⁻, q) · (1 − FSRisk(B⁻, q))
三个无监督代理：

| 因子 | 含义 |
|------|------|
| Trust | pitfall 被多条独立 rollout 支持的次数越多，越可信 |
| Locality | pitfall 与当前题目的语义相关性（embedding 相似度） |
| FSRisk（误抑制风险） | pitfall 与高一致答案簇推理是否冲突；pilot 解码中负分支是否降低答案一致性 |

当 Trust × Locality × (1−FSRisk) ≈ 0 时，α_j → 0，等价于 不做负向抑制，避免 Instruct 等模型上“错杀”正确推理。
特点：不主要优化采样效率，而是解决负经验不可靠时的安全性；适合负证据质量参差不齐的场景。

### BambooQA 主结果

#### Qwen3-30B-A3B-Thinking-2507

| Method | mean iter0 | mean iter1 | mean iter2 | mean iter3 | run1 iter0 | run1 iter1 | run1 iter2 | run1 iter3 | run2 iter0 | run2 iter1 | run2 iter2 | run2 iter3 | run3 iter0 | run3 iter1 | run3 iter2 | run3 iter3 |
|--------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| RSE | 61.6 | 56.3±4.7 | 50.7±2.6 | 48.8±1.1 | 60.8 | 59.2 | 53.6 | 49.6 | 60.0 | 49.6 | 47.2 | 47.2 | 64.0 | 60.0 | 51.2 | 49.6 |
| CTO | 59.7 | 37.9±1.8 | 36.7±1.2 | 38.5±4.7 | 60.8 | 36.1 | 35.2 | 31.9 | 55.2 | 37.4 | 36.6 | 41.3 | 63.2 | 40.3 | 38.2 | 42.3 |
| RG-CTO | 61.6 | — | — | — | 60.8 | — | — | — | 60.0 | — | — | — | 64.0 | — | — | — |
| Faster CTO | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

#### Qwen3-4B-Thinking-2507

| Method | mean iter0 | mean iter1 | mean iter2 | mean iter3 | run1 iter0 | run1 iter1 | run1 iter2 | run1 iter3 | run2 iter0 | run2 iter1 | run2 iter2 | run2 iter3 | run3 iter0 | run3 iter1 | run3 iter2 | run3 iter3 |
|--------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| RSE | 44.3 | 29.6±4.9 | 28.0±3.6 | 28.5±2.9 | 38.4 | 30.4 | 28.8 | 28.8 | 47.2 | 35.2 | 32.0 | 32.0 | 47.2 | 23.2 | 23.2 | 24.8 |
| CTO | 44.3 | — | — | — | 34.4 | 41.0 | 36.9 | 35.3 | — | — | — | — | — | — | — | — |
| RG-CTO | 44.3 | — | — | — | 38.4 | — | — | — | 47.2 | — | — | — | 47.2 | — | — | — |
| Faster CTO | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

#### Qwen3-4B-Instruct-2507

| Method | mean iter0 | mean iter1 | mean iter2 | mean iter3 | run1 iter0 | run1 iter1 | run1 iter2 | run1 iter3 | run2 iter0 | run2 iter1 | run2 iter2 | run2 iter3 | run3 iter0 | run3 iter1 | run3 iter2 | run3 iter3 |
|--------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| RSE | 41.1 | 40.0±1.3 | 33.9±2.1 | 33.1±2.1 | 41.1 | 41.6 | 36.8 | 36.0 | 41.1 | 38.4 | 32.0 | 32.0 | 41.1 | 40.0 | 32.8 | 31.2 |
| CTO | 41.1 | 36.3±0.8 | 33.6±0.7 | 34.0±1.8 | 50.4 | 35.2 | 32.8 | 32.0 | 48.8 | 36.8 | 33.6 | 36.4 | 41.1 | 36.8 | 34.4 | 33.6 |
| RG-CTO | 41.1 | — | — | — | 41.1 | — | — | — | 41.1 | — | — | — | 41.1 | — | — | — |
| Faster CTO | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

#### Phi-4-Reasoning

| Method | mean iter0 | mean iter1 | mean iter2 | mean iter3 | run1 iter0 | run1 iter1 | run1 iter2 | run1 iter3 | run2 iter0 | run2 iter1 | run2 iter2 | run2 iter3 | run3 iter0 | run3 iter1 | run3 iter2 | run3 iter3 |
|--------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| RSE | 35.7 | 35.7±2.1 | 38.1±1.6 | 37.6±2.0 | 35.7 | 36.8 | 36.0 | 35.2 | 35.7 | 37.6 | 40.0 | 37.6 | 35.7 | 32.8 | 38.4 | 40.0 |
| CTO | 35.7 | 34.4±2.4 | 38.7±3.7 | 38.4±4.1 | 39.2 | 31.2 | 33.6 | 34.4 | 35.7 | 35.2 | 42.4 | 44.0 | 35.7 | 36.8 | 40.0 | 36.8 |
| RG-CTO | 35.7 | — | — | — | 35.7 | — | — | — | 35.7 | — | — | — | 35.7 | — | — | — |
| Faster CTO | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

Contrastive Trajectory Optimization: Suppression-Safe Experience Control for Test-Time Reasoning
- 正文只保留 CTO 与一个可靠性扩展 Risk-Gated CTO（RG-CTO）。
- 将现有的 TR-CTO 统一改名为 RG-CTO；除非真正加入显式 KL 约束，否则不再使用 “Trust-Region” 命名。
- CTO-Wave 仅作为效率扩展放入实验或附录，不作为独立核心贡献。
- CTO-Rescore++ 及其与 Wave、TR-CTO 的组合不进入正文主线，避免将论文改成多个工程变体的堆叠。
- 最终 headline protocol 统一为 Iter0–Iter3：Iter0 为无经验采样，Iter1–Iter3 为三个经验引导轮次。旧的 Iter0–Iter2 消融结果必须明确标记为短周期诊断实验，不能与最终主结果混合统计。
Introduction
背景 说明 inference-time scaling / repeated sampling / self-improvement 方法能够通过多次 rollout 获得大量中间发现、失败模式和局部解法。问题在于，多数方法只将这些历史信息重新压缩进 prompt 或 memory，模型是否真的依赖这些经验并不受控制。这一段只需要建立“experience recycling 有价值，但 prompt-level utilization 不充分”的背景，不展开具体实验。
CTO 的核心思想 引出 CTO：
- 从当前问题的历史 rollout 中提取：positive propositions；critical pitfalls。
- 分别构造 positive / negative branch；
- 在相同 continuation prefix 下计算两个 token distribution；
- 用 contrastive score 显式增强与正经验一致、同时不被负经验支持的 continuation。
核心公式：
$$s_i
=
\ell_{\mathrm{pos},i}
-
\alpha
\ell_{\mathrm{neg},i}.$$
强调 CTO 相比普通 experience prompting 的关键差异：experience 不只是 prompt 中的自然语言建议，而是直接参与下一 token 的概率重构。这里不要讨论 Extraction / Source / Ranking 三模块。
关键问题——negative experience 并不天然 suppression-safe 指出已有结果已经显示：
- CTO 在部分 Qwen Thinking 设置中优于普通 experience recycling；
- 但在 Qwen Instruct、部分难题或跨模型设置中会发生明显退化；
- 这并不一定意味着 negative experience 在语义上完全错误；
- 更关键的问题是：某个 pitfall 即使描述了一个真实失败模式，也可能包含会出现在正确推理中的局部 token / reasoning pattern。
因此固定的$$\alpha=\alpha_0$$会把“negative experience 是否可信”与“应该以多强力度抑制它”混为一谈。把问题正式概括为：descriptive correctness does not imply suppression safety.
CG-CTO + contributions CG-CTO 对每个 guided round 的 negative experience 估计可靠性，并根据 support，locality，conflict risk 计算 gate，决定：
1. 哪些 negative items 应该被过滤；
2. 剩余 negative branch 应该以多强的 $$\alpha$$ 参与 contrastive decoding。
建议最终贡献只保留三条：
1. CTO formulation. 提出一种无需参数更新的 test-time experience utilization 方法，将当前问题历史 rollout 中提炼出的正负经验直接转化为 token-level contrastive decoding signal。
2. Suppression-risk diagnosis. 揭示 vanilla CTO 的主要失效模式：negative experience 的语义正确性不足以保证其适合作为 anti-expert，固定强度 suppression 可能导致 false suppression 和 negative transfer。
3. CG-CTO. 提出基于 support、locality 与 conflict risk 的 confidence-gated contrastive control，在 Qwen 数学推理上保持或改善 CTO 的收益，并在异模型 QA 设置中验证方法的可迁移性。
Related Work
Test-Time Scaling and Experience Recycling
分析
- repeated sampling / self-consistency / search；
- trajectory recycling；
- experience / reasoning memory；
- self-refinement。
目的只有一个：说明已有方法已经开始利用历史 rollout，但主要作用在 prompt、memory 或 trajectory selection 层面。
正文优先保留与本文最相关的少量工作，例如：
- Self-Consistency Improves Chain of Thought Reasoning in Language Models
- Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters
- Large Language Monkeys: Scaling Inference Compute with Repeated Sampling
- Do Not Waste Your Rollouts: Recycling Search Experience for Efficient Test-Time Scaling
- ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory
Experience Utilization and Self-Evolving Reasoning
讨论经验系统从“存什么”逐渐转向：
- 何时使用；
- 使用什么；
- 当前经验是否与 query / local state 匹配。
这里用来承接 CG-CTO 的 reliability / locality 思想。
可保留少量最相关工作：
- ExpeL: LLM Agents Are Experiential Learners
- Rethinking Experience Utilization in Self-Evolving Language Model Agents
- ExpSeek: Self-Triggered Experience Seeking for Web Agents
- Mem-$$\pi$$: Adaptive Memory through Learning When and What to Generate
- HippoSpark: An On-Demand Experience System for LLM Reasoning
不要在正文把它们展开成一整套 experience taxonomy。
Contrastive / Distribution-Level Decoding
这一段需要明确 CTO 真正的新颖性在哪里。
可讨论：
- DExperts；
- Contrastive Decoding；
- Context-Aware Decoding；
- Proxy-Tuning；
- reasoning-oriented contrastive decoding。
然后明确：CTO 的 novelty 不是简单使用 $$p_{\text{pos}}-\alpha p_{\text{neg}}$$ 这一数学形式，而是 positive / negative distribution 由当前问题自身的 rollout history 在线构造，并被迭代写回 reasoning process。CG-CTO 则进一步回答：online constructed anti-expert 何时值得信任。
Method
Problem Setup: Iterative Test-Time Experience Recycling
给定问题$$q$$，执行固定的多轮 test-time search：
- Iter0：无历史经验，标准 sampling；
- Iter1–Iter3：仅使用之前轮次产生的轨迹和经验；
- 每轮每题固定 $$R=32$$ 条 rollout；
- backbone 参数始终冻结。
记第$$r$$轮 rollout：
$$\mathcal{Y}^{(r)}
=
\{y_i^{(r)}\}_{i=1}^{R}.$$
经验库只能使用过去轮次：
$$\mathcal{B}^{(r)}
=
\operatorname{Merge}
\left(
\bigcup_{t<r}\mathcal{E}^{(t)}
\right).$$
这一节重点解释 protocol，避免 reviewer 误解为训练或使用未来信息。
From Rollouts to Positive and Negative Experience
使用同一个 backbone 对历史 rollout 做结构化 self-extraction：
$$D_\theta(q,y_i)
\rightarrow
(E_{i,+},E_{i,-}).$$
其中：
- $$E_{i,+}$$：verified propositions，包括中间结论、约束、定义、局部求解依据；
- $$E_{i,-}$$：critical pitfalls，包括错误假设、无效变换、失败分支和误导策略。
必须明确：
- 不使用 gold answer；
- 不使用 reference solution；
- extractor 只能看到当前问题、rollout 和其自身生成内容；
- malformed extraction 丢弃；
- 当前轮经验不能回流到当前轮，只能用于下一轮。
这里仅把 Extraction / Selection / Ranking 当作实现步骤，不是论文贡献：
$$\mathcal{B}^{(r)}
\xrightarrow{\text{select / rank / truncate}}
(\mathcal{R}^{(r)}_+,\mathcal{R}^{(r)}_-).$$
正文只写最终冻结配置，不再分别展开三套 design space。
建议主方法固定：
- same-problem experience；
- 一个冻结的 retrieval / ranking strategy；
- 固定最大经验数与 context budget。
如果已有实验最终证明 simple ordering 与 reranker 差异不稳定，就直接选择最简单、最稳定、最便于复现的版本作为默认值。
CTO: Experience-to-Logit Contrastive Decoding
把正负经验分别渲染为：
$$P_{\mathrm{pos}}
=
T_{\mathrm{pos}}(q,\mathcal{R}_+),
\qquad
P_{\mathrm{neg}}
=
T_{\mathrm{neg}}(q,\mathcal{R}_-).$$
在同一个 continuation prefix $$x_{<j}$$ 上分别计算：
$$\ell_{\mathrm{pos}}^{(j)}
=
\log p_\theta(\cdot\mid P_{\mathrm{pos}},x_{<j}),$$
$$\ell_{\mathrm{neg}}^{(j)}
=
\log p_\theta(\cdot\mid P_{\mathrm{neg}},x_{<j}).$$
只对 positive branch 的 top-$$K_{\mathrm{cto}}$$ token 允许 contrastive intervention：
$$\mathcal{S}^{(j)}
=
\operatorname{TopK}
(\ell_{\mathrm{pos}}^{(j)},K_{\mathrm{cto}}).$$
Vanilla CTO：
$$s_i^{(j)}
=
\begin{cases}
\ell_{\mathrm{pos},i}^{(j)}
-
\alpha_0
\ell_{\mathrm{neg},i}^{(j)},
&
i\in\mathcal{S}^{(j)},\\
\ell_{\mathrm{pos},i}^{(j)},
&
\text{otherwise}.
\end{cases}$$
正文说明：
- $$\alpha_0$$ 和 $$K_{\mathrm{cto}}$$ 在 development setting 上冻结；
- positive / negative branch 不分别生成两条轨迹；
- 只从最终融合后的 distribution 中采样一次。
Why Fixed Negative Suppression Can Fail
不要再放一大套 faithfulness 分析。只解释一个核心 failure mode：对于 negative item $$e$$，即使它描述的 failure mode 本身合理，也可能出现：
1. 只由单条偶然 rollout 支持；
2. 与当前问题局部状态相关性低；
3. 与高一致 positive evidence 冲突。
因此：
$$\text{semantic validity}(e)
\not\Rightarrow
\text{safe-to-suppress}(e).$$
这正是 CG-CTO 要解决的问题。
3.5 CG-CTO: Confidence-Gated Negative Control
对每个 negative experience $$e$$定义：
$$w(e)
=
\operatorname{clip}
\left[
u(e)
\cdot
l(e,q)
\cdot
(1-c(e)),
0,1
\right].$$
其中：
- $$u(e)$$：support，衡量该 pitfall 是否得到多条独立 rollout / answer cluster 支持；
- $$l(e,q)$$：locality，衡量其与当前问题及局部求解状态的相关性；
- $$c(e)$$：conflict risk，衡量其与 positive propositions 或高一致推理模式的冲突程度。
先过滤$$w(e)<\delta$$的 negative item，对剩余 negative items 聚合得到 round-level gate：
$$g^{(r)}
=
\frac{1}{|\mathcal{R}_{-}^{(r)}|}
\sum_{e\in\mathcal{R}_{-}^{(r)}}w(e).$$
有效 suppression coefficient：
$$\alpha_r
=
\alpha_0 g^{(r)}.$$
最终：
$$s_i^{(j)}
=
\begin{cases}
\ell_{\mathrm{pos},i}^{(j)}
-
\alpha_r
\ell_{\mathrm{neg},i}^{(j)},
&
i\in\mathcal{S}^{(j)},\\
\ell_{\mathrm{pos},i}^{(j)},
&
\text{otherwise}.
\end{cases}$$
Experiment
Setting
Benchmarks 数学主实验：HMMT24；HMMT25；HLE-Math-text。
Qwen Backbones 建议正文优先：
1. Qwen3-4B-Thinking-2507：主 setting；
2. Qwen3-4B-Instruct-2507：关键 failure / robustness setting；
3. Qwen3-30B-A3B-Thinking-2507：scale check。
如果 30B 新实验成本过高，可只保留已有、协议完全一致的 HMMT24 结果；否则放附录。
Transfer Setting 只补一组 Phi-4-Reasoning + BambooQA，作用仅是证明：CG-CTO 并非只依赖 Qwen chat template 或数学 benchmark。不需要再增加第二个 QA、coding 或 agent benchmark。
Baselines 正文只比较：Standard Sampling / Iter0；RSE；CTO；CG-CTO。
Protocol 统一：
- Iter0–Iter2；
- 每轮 32 rollouts；
- 3 seeds；
- 所有直接比较方法共享同 seed 的 Iter0；
- 相同 sampling hyperparameters；
- 相同 answer parser / evaluator；
- $$\alpha_0$$、$$K_{\mathrm{cto}}$$、$$\delta$$只在独立 development setting 上冻结一次。

### Results

#### HMMT24 / HMMT25 / HLE-Math-text 主结果（Pass@1，%）

| Model | Method | HMMT24 iter0 | HMMT24 iter1 | HMMT24 iter2 | HMMT25 iter0 | HMMT25 iter1 | HMMT25 iter2 | HLE iter0 | HLE iter1 | HLE iter2 |
|-------|--------|-------------|-------------|-------------|-------------|-------------|-------------|----------|----------|----------|
| Qwen3-30B-A3B-Thinking-2507 | RSE | 55.8 | 68.8±2.7 | 70.6±3.4 | 66.2 | 66.7 | **80.0** | 23.4 | 38.8±1.4 | 40.8±2.0 |
| Qwen3-30B-A3B-Thinking-2507 | CTO | 55.8 | **73.6±1.7** | **72.8±1.9** | 66.2 | **79.1** | **80.0** | 23.4 | 36.9±1.0 | 37.1±0.5 |
| Qwen3-30B-A3B-Thinking-2507 | RG-CTO | 55.8 | — | — | 66.2 | — | — | 23.4 | — | — |
| Qwen3-4B-Thinking-2507 | RSE | 43.2 | 58.4±4.6 | 59.1±3.6 | 54.3 | **68.6** | **72.2** | 13.6 | **18.5±2.5** | **20.0±4.2** |
| Qwen3-4B-Thinking-2507 | CTO | 43.2 | 60.0±3.1 | **62.2±3.2** | 54.3 | 66.5 | 68.4 | 13.6 | 17.7±2.5 | 17.8±2.2 |
| Qwen3-4B-Thinking-2507 | RG-CTO | 43.2 | — | — | 54.3 | — | — | 13.6 | — | — |
| Qwen3-4B-Instruct-2507 | RSE | 25.2 | **39.6±0.6** | **42.3±1.7** | 21.8 | **29.6** | **30.1** | 8.7 | **8.6±0.2** | **9.4±1.0** |
| Qwen3-4B-Instruct-2507 | CTO | 25.2 | 29.1±1.5 | 30.3±2.7 | 21.8 | 27.7 | 28.3 | 8.7 | 6.9±1.7 | 6.9±1.2 |
| Qwen3-4B-Instruct-2507 | RG-CTO | 25.2 | — | — | 21.8 | — | — | 8.7 | — | — |
| Phi-4-Reasoning | RSE | 28.0 | **28.3±1.0** | **28.4±0.4** | 21.7 | — | — | 5.9 | **6.0±0.5** | **5.9±0.2** |
| Phi-4-Reasoning | CTO | 28.0 | 12.3±1.2 | 12.8±0.5 | — | — | — | 5.9 | 6.9±1.7 | 6.9±1.2 |
| Phi-4-Reasoning | RG-CTO | 28.0 | — | — | — | — | — | 5.9 | — | — |

> 注：iter0 为所有方法共享的 baseline rollout。HMMT24 / HLE 的 iter1/iter2 为 3-seed 均值 ± 标准差（RSE 来自 `RSE/runs/reproduce/`，CTO 来自 3-run vllm）。HMMT25 Thinking 的 CTO 为 CTO-base（`nocross_llm_judge`）单 run；RSE 为同协议对照实验（30B/4B 来自 CTO 论文主表，Instruct 来自附录）。Phi-4 HMMT25 仅有 RSE iter0（21.7%）。RG-CTO 待跑。

## Iter0 基线结果（已复制到 `results/iter0/`）

所有方法共享同一 iter0 rollout（RSE reproduce run0；BambooQA 取各模型最佳单 run）。Pass@1 汇总见 `results/summaries/iter0_pass_at1.json`。

| 数据集 | 模型 | iter0 Pass@1 | 来源 |
|--------|------|-------------|------|
| HMMT24 | Qwen3-30B-Thinking | 55.4 | RSE reproduce run0 |
| HMMT24 | Qwen3-4B-Thinking | 43.3 | RSE reproduce run0 |
| HMMT24 | Qwen3-4B-Instruct | 25.6 | RSE reproduce run0 |
| HMMT24 | Phi-4-Reasoning | 28.0 | RSE reproduce run0 |
| HMMT25 | Qwen3-30B-Thinking | 66.2 | CTO step1 |
| HMMT25 | Qwen3-4B-Thinking | 54.3 | CTO step1 |
| HLE-Math-text | Qwen3-30B-Thinking | 23.4 | RSE reproduce run0 |
| HLE-Math-text | Qwen3-4B-Thinking | 13.5 | RSE reproduce run0 |
| HLE-Math-text | Qwen3-4B-Instruct | 8.3 | RSE reproduce run0 |
| HLE-Math-text | Phi-4-Reasoning | 6.6 | RSE reproduce run0 |
| BambooQA | Qwen3-30B-Thinking | 64.0 | RSE run3（最佳单 run） |
| BambooQA | Qwen3-4B-Thinking | 44.8 | CTO run1 |
| BambooQA | Qwen3-4B-Instruct | 50.4 | CTO run1 |
| BambooQA | Phi-4-Reasoning | 39.2 | CTO run1 |

只分析三件事：
A. CTO 是否真的比普通 experience recycling 更能利用经验
重点看 HMMT24 / Thinking。不要再宣称 CTO 在所有 benchmark 都稳定提升，因为现有 HLE / Instruct 已经不支持这一结论。
B. Vanilla CTO 的主要问题是否是负向抑制过强
重点看：
- HMMT24 / Instruct；
- HLE-Math-text / Thinking。
如果 CG-CTO 相比 CTO 显著恢复性能，同时不破坏 HMMT24 / Thinking 上的正收益，就形成完整的核心证据。
C. CG-CTO 是“安全修正”而不是追求所有 setting SOTA
论文目标应该表述为：CG-CTO improves the robustness of contrastive experience utilization across heterogeneous reasoning settings.
而不是：CG-CTO universally improves reasoning.
不需要分析什么
主结果之后不要再展开：
- 不同 extractor 哪个最好；
- Local-Seq / Local-Retrieval 谁最好；
- embedding / reranking 谁最好；
- 哪个 cluster strategy 最好。
这些不再属于 headline claim。
Ablation
