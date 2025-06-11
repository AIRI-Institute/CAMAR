## A Standardised Performance Evaluation Protocol Suite for Continuous Multi‐Agent Pathfinding

We adapt ["A Standardised Performance Evaluation Protocol for Cooperative MARL"](https://arxiv.org/pdf/2209.10485) to evaluate continuous multi‐agent pathfinding in CAMAR. In particular, we want to measure generalisation along two axes:

1. **Number of agents**
2. **Map structure/difficulty**

Below, we **(a)** adopt most of the original default parameters, and **(b)** introduce three "difficulty tiers" of evaluation protocols — **[Easy](#31-protocol-easy--generalisation-to-unseen-task-seeds), [Medium](#32-protocol-medium--generalisation-to-new-maps-of-similar-difficulty), [Hard](#33-protocol-hard--generalisation-to-unseen-map-families--number-of-agents)** — each targeting a different notion of generalisation.

Here is a **short overview** of all three protocols. For full details, see the sections below.

|                                        | **Easy**                                                                                                                                                                        | **Medium**                                                                                                                                                                                                                                     | **Hard**                                                                                                                                                               |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Why**                                | Test that the method can solve the problem without testing generalisation.                                                                                                      | Test that the method can solve the problem *and* generalise across similar map types including varying number of agents.                                                                                                                                                          | Test that the method can generalise to near-real-world settings.                                                                                                       |
| **How to train & evaluate**            | **Train on** `random_grid_h20_w20_a8_o0` (see [`protocols/easy-medium/`](./easy-medium/))<br>**Evaluate on** the same `random_grid_h20_w20_a8_o0`<br>**Repeat** for all 12 maps | **Train on** `random_grid_h20_w20_a8_o0` (see [`protocols/easy-medium/`](./easy-medium/))<br>**Evaluate on** **all 12** maps (not only the number of agents used for training)<br> **Repeat** for all 12 maps | **Train on** any map excluding MovingAI<br>**Evaluate on** MovingAI street collection ([`protocols/hard/eval/street/`](./hard/eval/street/)) with varying agent counts |
| **Total number of models/evaluations** | 12 trained models, 12 evaluations (each 1K episodes)                                                                                                                           | 12 trained models, 144 evaluations (each 1K episodes)                                                                                                                                                                                          | 1 trained model, TODO evaluations (each 1K episodes).                                                                                                                |
| **What to report**                     | `Final scores` - mandatory<br>`sample-efficiency curves` - strongly recommended                                                                                                 | `Final scores` - mandatory<br>`sample-efficiency curves` - recommended                                                                                                                                                                         | `Final scores` - mandatory<br>`metrics vs num_agents` - mandatory<br>`sample-efficiency curves` - if resources allow                                                   |

1. Include comparisons with other methods on the same charts/tables.
2. Make sure data for all charts and tables are publicly available (e.g. as CSVs in a GitHub repo or supplementary).

---

> **Input:**
>
> *   A family of maps $\mathcal{M}$.
> *   For each map $m \in \mathcal{M}$, a set of pathfinding tasks $\mathcal{T}_m$. Each task $t \in \mathcal{T}_m$ consists of start/goal locations for all agents.
> *   A suite of algorithms $\mathcal{A}$.

### 1. Evaluation Parameters – Defaults

1. **Training Timesteps $T$.**

   *  Off‐policy methods: $T = 2M$ timesteps.
   *  On‐policy methods: $T = 20M$ timesteps.

2. **Independent Training Runs $R$.**

   *  $R = 3$ independent seeds/runs.

3. **Evaluation Episodes per Interval $E$.**

   *  $E = 1K$ episodes.
   *  *(Since CAMAR is fully vectorised, running 1 000 environments in parallel gives almost the best SPS)*

4. **Evaluation Intervals $\mathcal{I}$.**

   *  Off‐policy: every $10K$ training timesteps.
   *  On‐policy: every $100K$ training timesteps.

---

### 2. Performance & Uncertainty Quantification

1. **Performance Metrics**

   *  **Training Return** $G$ *(higher is better)*: the cumulative scalar reward used by the RL algorithm (only for sample efficiency charts).

   *  **Success Rate** $SR$ *(higher is better)*: fraction of agents that reached their goal by episode’s end.

   *  **Flowtime** $FT$ *(lower is better)*: sum of each agent’s path length (measured in simulation time).

   *  **Makespan** $MS$ *(lower is better)*: maximum over all agents’ path lengths (measured in simulation time).

   *  **Coordination** $CO$ *(higher is better)*: The inverted number of collisions per agent.

2. **Per‐Task Evaluation**

   (This is optional in CAMAR since each map can generate a very large amount of tasks)

   *  For each task $t$, algorithm $a$, and evaluation interval $i$:

     1. Run $E$ episodes on task $t$ (using eval_keys).
     2. Compute the mean return $G_t^a$ over those $E$ episodes, along with a 95% confidence interval.

3. **Per‐Protocol/Environment Evaluation** <a name="2_3_per_protocol_eval"></a>

   At each evaluation interval $i$:

   1. **Normalized Absolute Return.**

      *  Collect the best joint policy checkpoint found so far for each training run $r$.
      *  For each algorithm $a$, run $E$ episodes on *every* task $t\in T$ (across all maps specified by the protocol).
      *  Compute the normalized absolute return as the mean return of all evaluation episodes using the best joint policy found during training:

      $G_{norm} \left( a,r,t \right) = \frac{G_{t}^{a,r} - \min_{a',r',t'} G_{t'}^{a',r'}}{\max_{a',r',t'} G_{t'}^{a',r'} - \min_{a',r',t'} G_{t'}^{a',r'}}$

      *  For each algorithm $a$, form an evaluation matrix $\{G_{t}^{a,r}\}$ with shape $(R,|T|)$.
   2. **Aggregated Statistics.**

      *  Compute the Interquartile Mean (IQM) and Optimality Gap with 95% stratified bootstrap CIs.
      *  Compute probability of improvement and performance profiles.
      *  Optionally plot **Sample Efficiency Curves**: track IQM of the normalized return as a function of total timesteps with shaded areas indicating 95% CI.

---

### 3. Specific Evaluation Protocols

> **Note on JAX random seeds:**
> For all evaluation episodes (Easy, Medium, Hard), use exactly the same set of JAX seeds:
>
> ```python
> test_keys = jax.random.split(jax.random.key(5), num_episodes)
> ```
>
> That is, set `seed=5` once, then split into `num_episodes` evaluation keys. This ensures reproducibility and fair comparison of different algorithms

Below, each sub‐section describes one "difficulty tier." In each case, we specify which maps are in the training set and which in the test and how do testing procedure and result aggreagtions.

---

#### 3.1 Protocol "Easy" – Generalisation to Unseen Task Seeds

1. **Objective**
   Measure how well an algorithm trained on one set of random seeds (start/goal configurations) generalises to *new seeds* on the *same map class* and *same number of agents*.

2. **Training**

   1. Fix a map class $m$ ∈ {`labmaze_grid`, `random_grid`}
   2. Fix the number of agents $N \in$ { $8$, $32$ }
   3. Fix obstacle parameters (`obstacle_density` or `extra_connection_probability`)
   4. Train on maps from `protocols/easy/` (already contains all necessary settings and can be used for the environment creation as is); in total, train on 12 different settings $(m, N, o_{params})$.

3. **Testing**

   1. For each of those 12 trained models, evaluate on *new* JAX seeds (using `test_keys`). Test on the *same setting* that were used during training, you'll get 12 evaluation in total (i.e., "train MAPPO on `labmaze_grid_Y`" for 20M steps, "test on `labmaze_grid_Y` with `test_keys`")
   2. Each evaluation interval: run $E=1000$ episodes for each $(m, N, o_{params})$.

4. **Aggregation** (How to present final scores) <a name="easy_agg"></a>

   1. For the final score, aggregate metrics on each map type separately as specified in [Section 2](#2-performance--uncertainty-quantification).
   2. You will get $5$ metrics ($G, SR, FT, MS, CO$ for every map $m$) for every map $m$. Aggregate all evaluations on `random_grid` and `labmaze_grid` (mandatory).
   3. Optionally, include additional charts and analysis as in [Section 2.3](#2_3_per_protocol_eval) (can be placed in an appendix)

> **Note:** `random_grid` does not guarantee full connectivity, so it is acceptable if $SR<1.0$ on some seeds.

---

#### 3.2 Protocol "Medium" – Generalisation to New Maps of Similar Difficulty

1. **Objective**
   Evaluate how well an agent generalise to *both seen and unseen maps*.

2. **Training**

   Exactly the same 12 $(m, N, o_{params})$ settings that Protocol Easy use. We again train 12 models.

3. **Testing**

   1. Each of the models test on all settings in [`protocols/easy-medium/`](./easy-medium/) regardless if whether $o_{params}$ or number of agents were seen during training.

4. **Aggregation**

   1. Almost the same procedure as in [Section 3.1. Aggregation](#easy_agg).
   2. Now for each map $m$ ∈ {`labmaze_grid`, `random_grid`} we aggargate statistics from total 12 models each trained on the separate map type, so each setting will be evaluated on 12 models. Both `random_grid` and `labmaze_grid` get $144K$ evaluation episodes each
   3. Optionally extend the charts from [Section 3.1. Aggregation.3](#easy_agg) to show performance on each $o_{params}$ or number of agents separately. It can be useful because models trained without obstacles are now evaluated on maps with obstacles. *In CAMAR's default observation design, agents do not distinguish between other agents and obstacles, so if an agent learns to avoid collisions with other agents, it may also learn to avoid obstacles - and we test that*

---

#### 3.3 Protocol "Hard" – Generalisation to Unseen Map Families & Number of Agents

1. **Objective**
   Measure strong generalisation by training on *any maps* excluding the MovingAI 2D Benchmark maps, then testing — *with no further fine‐tuning* — on fully unseen, real-world-style maps from the MovingAI collection. This also assesses how performance scales when the number of agents during testing differs from training.

2. **Training**

   1. Choose any set of maps. These maps must **exclude** any MovingAI 2D maps.
   2. Select any number of agents for training.
   3. Because MovingAI maps are large and complex, it is acceptable not to evaluate every 10K or 100K steps during training if resources are limited. In this case, only final scores are mandatory (these are the main metrics for planning tasks).

3. **Testing**

   1. Prepared a **held-out set of MovingAI maps**:

      *  Every city from the [MovingAI street collection](https://movingai.com/benchmarks/street/index.html).
      *  CAMAR configurations for these maps are in `protocols/hard/test/` - with several configurations for different number of agents.
      * Optionally you can include other collections (i.e. bgmaps), but this can be computationally expensive.

   3. You do **not** train separate models for each agent‐count; the single model architecture should handle variable $N$ (for example each agent can use the same policy model in a decentralized MARL).

4. **Aggregation**

   1. Evaluate the model on each collection from the MovingAI separately.
   2. You must include plots that show how performance scales with the number of agents (include as many agents in evaluation as your model can handle). Plot IQM with shaded area indicating 95% CI. Comparisons with methods or modifications is preferred.
   3. Specify in your results how many agents the model was trained with and compare its performance when tested with more agents.
   4. Present the final results as **4 charts**:
      *  `SR vs num_agents`
      *  `FT vs num_agents`
      *  `MS vs num_agents`
      *  `CO vs num_agents`

---

### 4. Reporting

Below is a concise checklist of everything you should include when reporting results for the Easy, Medium, and Hard protocols. The goal is to be brief and to the point—save detailed examples or extended discussions for appendices or supplementary materials.

---

#### 4.1 Experimental Details (Mandatory)

* **Hyperparameters & Architectures**
  *   List all learning‐rate schedules, discount factors, network sizes, replay‐buffer settings, and any model‐specific tweaks (e.g. multi‐step returns, gradient clipping).
  *   Describe the neural network architecture (layers, activations, **weight sharing**, etc.).

* **Environment & Map Configurations**
  *   Provide (or link to) the exact map‐generation settings used for Easy, Medium, and Hard (if these differ from those specified above).

* **Compute & Runtime Summary**
  *   Specify hardware (e.g. GPU type) and approximate wall‐clock training time per run.

---

#### 4.2 Key Results (Mandatory)

*  **Primary Metrics**
  *   For each $(m, o_{params}, N)$ group in Easy and Medium, and for each MovingAI collection in Hard, report:
  – Interquartile Mean (IQM) ± 95 % CI of all metric (Normalized Returns, Success Rate, Flowtime, Makespan, Coordination) as specified in [3.1](#31-protocol-easy--generalisation-to-unseen-task-seeds), [3.2](#32-protocol-medium--generalisation-to-new-maps-of-similar-difficulty), [3.3](#33-protocol-hard--generalisation-to-unseen-map-families--number-of-agents).

*  **Scaling Curves**
  *   In Hard, show how SR, FT, MS, and CO change as you increase the number of agents (e.g. `SR vs num_agents`, etc.). These four charts are required. See [Protocol "Hard"](#33-protocol-hard--generalisation-to-unseen-map-families--number-of-agents) for details.

*  **Optional Analysis**
  *   To show generalisation over other parameters (i.e. obstacle density), you may add an additional figure of each metric vs custom_parameter.
  *   Sample-efficiency plots can be included in appendix or supplementary.

---

#### 4.3 Additional Materials (Optional)

  *   Make sure all data used for all plots and tables are publicly available (for testing reproducibility and results sharing).
  *   If training sometimes fails at high agent counts or certain densities, mention it briefly and suggest a possible reason.
