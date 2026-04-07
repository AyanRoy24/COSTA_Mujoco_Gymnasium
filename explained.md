You can think of COSTA as a **3-stage pipeline**:

1) **Build offline datasets** (Safety-Starter-Agents)  
2) **Offline training (per-task models + meta context encoder)** (COSTA scripts)  
3) **Online adaptation / evaluation** (COSTA `online_adaptation*.py`)

Below is the “code flow” by entry point and how data/artifacts move.

> Note: my code search results are truncated (GitHub search API limit). If you want to inspect more matches directly in GitHub UI, use a repo search like:  
> `repo:ApocalypseX/COSTA "ex.add_config" "configs/"` in GitHub’s search bar.

---

## 0) Where configuration comes from (`configs/<task>.json`)
Most COSTA scripts do the same pattern:

- read JSON: `config_dict = read_json_dict("configs/"+args.task+".json")`
- adjust a few fields (paths, seed, etc.)
- pass the entire dict into **Sacred**: `ex.add_config(config_dict)`
- call `ex.run()`
- inside training functions, Sacred injects a scoped `_config` via `@ex.capture(prefix=...)`

Example config: `COSTA/configs/ant-dir-safe.json` includes:
- env/task sizes: `state_dim`, `action_dim`, `num_tasks`, etc.
- offline data path: `data_path`
- algorithm hyperparams: `cql_weight`, `cpq_weight`, `safety_threshold`, etc.
- meta encoder paths under `meta_params` (e.g., `mlp_attn_path`, `focal_path`, `corro_path`) citeturn3commentary554443599158931

---

## 1) Offline Data Collection (Safety-Starter-Agents)
This is the “producer” of the `.npz` that everything else consumes.

### Entry point
`safety-starter-agents/scripts/mujoco_experiment.py`

Flow (high-level):
- parse args: `--task ant-dir-safe --algo cpo --goal 0`
- loads default + merges JSON config: `config="configs/"+task+".json"` then loads JSON and `deep_update_dict(...)`
- creates env (`NormalizedBoxEnv(ENVS[...])`), `env.reset_task(goal)`
- runs chosen safe-RL algorithm (CPO etc.)
- periodically saves trajectories; final offline buffer ends up as:
  `.../offline_data/<task>_<goal>/offline_buffer.npz` (as README describes)

You can see the config merge + `reset_task(goal)` logic in `mujoco_experiment.py`. citeturn1commentary8747340318157228

**Artifact produced:**
- `offline_buffer.npz` containing (at least) arrays like observations/actions/next_obs/reward/cost.

---

## 2) Offline Training Stage A: train per-task CVAE (`run_vae.py`)
### Entry point
`COSTA/run_vae.py` citeturn2commentary9784034731005267

Flow:
1. CLI args: `--task`, `--goal`
2. Loads `configs/<task>.json`
3. Builds run directory with timestamp:
   - `vae_params["path"] = <base>/<task>_<goal>/vae/<timestamp>`
4. Points `vae_params["data_path"]` at:
   - `<base_data>/<task>_<goal>/offline_buffer.npz`
5. Sacred calls `run(_config)` with `prefix='vae_params'`:
   - creates `SimpleSafeReplayBuffer(...)`
   - `buffer.init_buffer(_config["data_path"])` loads the `.npz`
   - `data = buffer.sample_all()`
   - trains `CVAE(state_dim, action_dim, latent_dim)` using actions+observations
   - saves model checkpoints under `<path>/model/...`

**Artifact produced:**
- something like `run/<task>_<goal>/vae/<timestamp>/model/epoch*.pt` (your config also has a `vae_path` field used later).

---

## 3) Offline Training Stage B: train per-task cost dynamics / discriminator (`run_discriminator.py`)
### Entry point
`COSTA/run_discriminator.py` citeturn2commentary1454603274762754

Flow is similar to VAE:
1. Loads `configs/<task>.json`
2. Sets:
   - `discriminator_params["data_path"] = .../<task>_<goal>/offline_buffer.npz`
   - `discriminator_params["path"] = .../<task>_<goal>/discriminator/<timestamp>`
3. In `run(_config)`:
   - loads offline buffer
   - shuffles + train/test split
   - trains `EnsembleDiscriminatorModel(...)` with `(obs, action, next_obs) -> cost` style supervision
   - saves **best** model into something like:
     `run/<task>_<goal>/discriminator/best_model.pt` (later code loads exactly this filename)

**Artifact produced:**
- `run/<task>_<goal>/discriminator/best_model.pt`

---

## 4) Offline Training Stage C: train meta “context encoder” (`run_context_encoding.py`)
### Entry point
`COSTA/run_context_encoding.py` citeturn2commentary8801518589332659

This stage **depends on the discriminators you trained for each task**.

Flow:
1. Loads `configs/<task>.json`
2. Sets output path:
   - `context_params["path"] = <base>/<task>/context_encoder/<timestamp>`
3. In `run(_config)` with `prefix='context_params'`:
   - builds an encoder (notably `MLPAttnEncoder(...)`)
   - builds a decoder `Decoder(...)`
   - loads cost models for each training task:
     ```python
     for task in range(num_tasks):
         temp_model = torch.load("run/"+task_name+"_"+str(task)+"/discriminator/best_model.pt")
     ```
   - constructs `OfflineMetaLearner(...)` and calls `learner.train()`

**Artifact produced:**
- `.../context_encoder/<timestamp>/model/encoder100.pt` (README tells you to copy this relative path into config)

This is the file later consumed by meta-policy training:
`torch.load("run/"+env_name+"/context_encoder/"+ meta_params[encoder_type+"_path"])` in `run_meta_cpq.py`. citeturn2commentary5414312463841693

---

## 5) Offline Training Stage D: train the meta-policy (`run_meta_cpq.py`)
### Entry point
`COSTA/run_meta_cpq.py` citeturn2commentary5414312463841693

Flow (from the visible parts):
1. Loads `configs/<task>.json`
2. Builds log dir using `make_log_dirs(...)` (creates `run/<task>/<algo>/<seed_timestamp...>/`) citeturn3commentary2479094733090706
3. Sacred runs training, and code:
   - creates logger, logs hyperparams
   - **loads the context encoder checkpoint** based on config:
     ```python
     encoder = torch.load("run/"+env_name+"/context_encoder/"+ meta_params[encoder_type+"_path"])
     ```
   - creates a `MFMetaSafePolicyTrainer(...)`
   - calls `policy_trainer.train()`

**Artifact produced:**
- policy checkpoint (README example):
  `run/<task>/meta_cpq/mlp_attn/seed_0_timestamp_<...>/model/policy.pth`

(Other baselines do analogous things, e.g. `run_focal_cpq.py` loads `meta_params["focal_path"]` instead.) citeturn1commentary9206457119199385

---

## 6) Online Adaptation / Evaluation (`online_adaptation.py`)
### Entry point
`COSTA/online_adaptation.py` citeturn2commentary6444842735410941

Flow:
1. CLI args include:
   - `--policy_path` (defaults to some example path)
   - `--ood` for task generalization
   - ablation flags `--wo_iid`, `--wo_exp`
2. Loads `configs/<task>.json`, sets:
   - `config_dict["policy_path"] = args.policy_path`
   - `config_dict["ood"] = args.ood`
3. Chooses log directory name based on flags (`online_adaptation`, `task_generalization`, etc.)
4. Inside `do_adaptation()` (not fully shown in the snippet), it:
   - runs rollouts in env(s)
   - records reward + cost per episode
   - saves plots + numpy dump:
     - `adaptation_context.npy`
     - “adaptation reward/cost” figures
   - logs per-task mean reward/cost

You can see the end-of-loop logging + plotting + saving in the snippet. citeturn2commentary6444842735410941

---

## End-to-end artifact dependency graph (what must exist before what)

1. `offline_data/<task>_<goal>/offline_buffer.npz`
   → consumed by `run_vae.py` and `run_discriminator.py` citeturn2commentary9784034731005267turn2commentary1454603274762754

2. `run/<task>_<goal>/discriminator/best_model.pt`
   → consumed by `run_context_encoding.py` (loads one per goal/task index) citeturn2commentary8801518589332659

3. `run/<task>/context_encoder/<timestamp>/model/encoder100.pt` (path written into `configs/<task>.json` under `meta_params`)
   → consumed by `run_meta_cpq.py` citeturn2commentary5414312463841693turn3commentary554443599158931

4. `.../model/policy.pth` produced by `run_meta_cpq.py`
   → consumed by `online_adaptation.py` via `--policy_path` citeturn2commentary6444842735410941

---

### Clarifying question
When you say “explain the code flow”, do you want:
1) a **conceptual pipeline** (what I did above), or  
2) a **call-stack style walkthrough** (which classes/functions are called inside `OfflineMetaLearner.train()` and `MFMetaSafePolicyTrainer.train()`)?

If you want (2), tell me which path you care about most: **AntDir** (`ant-dir-safe`) or an OOD case like **CheetahVel**, and I’ll trace the trainer/learner internals next.

---

Trace for **AntDir** (`ant-dir-safe`) in this repo (commit `62208b592b14d31312983cd2e9615ac65873be99`). I’ll follow the *actual entrypoints + the exact tensors they build*, and show where the **context embedding `z`** comes from and how it gets appended to state.

---

## Stage 1 — Per-task CVAE training (`COSTA/run_vae.py`)
**Purpose:** learn a behavior prior / action model per task from offline data.

**Flow:**
1. CLI: `python run_vae.py --task ant-dir-safe --goal <k>`
2. Loads `configs/ant-dir-safe.json`, then rewrites:
   - `vae_params["data_path"] = <vae_params.data_path>/ant-dir-safe_<k>/offline_buffer.npz`
   - `vae_params["path"] = <vae_params.path>/ant-dir-safe_<k>/vae/<timestamp>` citeturn6commentary8102833508596245
3. `run(_config)` (Sacred with prefix `vae_params`):
   - `buffer = SimpleSafeReplayBuffer(...)`
   - `buffer.init_buffer(_config["data_path"])` loads the `.npz`
   - `data = buffer.sample_all()`
   - trains a `CVAE(state_dim, action_dim, latent_dim)` on:
     - **X = observations**
     - **Y = actions** (dataset is `(actions, observations)`) citeturn6commentary8102833508596245

**Artifact expected later:**
- The meta-policy code later loads VAEs from:
  `run/ant-dir-safe_<k>/vae/vae.pt` (note: this is a fixed filename in `run_meta_cpq.py`, so your VAE training code must save/copy to that path). citeturn6commentary8005533592196931

---

## Stage 2 — Per-task cost discriminator (`COSTA/run_discriminator.py`)
**Purpose:** train a per-task **cost dynamics / cost classifier** model used by the context-encoder training stage.

**Flow:**
1. CLI: `python run_discriminator.py --task ant-dir-safe --goal <k>`
2. Loads config and rewrites:
   - `discriminator_params["data_path"] = .../ant-dir-safe_<k>/offline_buffer.npz`
   - `discriminator_params["path"] = .../ant-dir-safe_<k>/discriminator/<timestamp>`
3. In `run(_config)`:
   - loads offline buffer
   - builds supervised dataset:
     - inputs: `(actions, observations, next_observations)`
     - label: `costs`
   - trains `EnsembleDiscriminatorModel(...)` citeturn6commentary4940024370337763

**Artifact expected later:**
- `run/ant-dir-safe_<k>/discriminator/best_model.pt` (this exact filename is loaded by `run_context_encoding.py`). citeturn6commentary9506004010723076

---

## Stage 3 — Train the **context encoder** (`COSTA/run_context_encoding.py` → `OfflineMetaLearner`)
**Purpose:** learn encoder `E(trajectory)` that maps **cost-trajectory context** to an embedding `z` (dimension `encode_dim`, e.g. 16).

### 3A) Entry point (`run_context_encoding.py`)
1. CLI: `python run_context_encoding.py --task ant-dir-safe`
2. Writes output dir:
   - `context_params["path"] = <base>/ant-dir-safe/context_encoder/<timestamp>`
3. Builds:
   - `encoder = MLPAttnEncoder(input_dim = 2*obs_dim + act_dim + 1, output_dim = encode_dim)`
     - that input dimension corresponds to concatenating:  
       **obs + act + next_obs + cost** (cost is 1 dim)
   - `decoder = Decoder(input_dim = 2*obs_dim + act_dim + encode_dim, output_dim=1)`
4. Loads discriminators for each training task:
   - `torch.load("run/ant-dir-safe_<k>/discriminator/best_model.pt")` for `k in [0..num_tasks-1]`
5. Creates `OfflineMetaLearner(...).train()` citeturn6commentary9506004010723076

### 3B) What `OfflineMetaLearner` does with AntDir buffers
In `OfflineMetaLearner.__init__`, it immediately calls `self.load_buffer()` which loads:
- `offline_data/ant-dir-safe_0/offline_buffer.npz`
- `offline_data/ant-dir-safe_1/offline_buffer.npz`
- `offline_data/ant-dir-safe_2/offline_buffer.npz`
into `self.multi_task_buffer[goal]`. citeturn7commentary2634388741968144

So the context encoder is trained on **all tasks’ offline buffers**.

> I can’t show you the full `train()` body because my code search view truncated and the “getfile” subcall failed for the rest of the file, but the key data interface is visible: the learner samples from `SimpleSafeReplayBuffer` per goal and builds context from offline rollouts. citeturn7commentary2634388741968144

**Artifact consumed later:**
- You copy the produced encoder checkpoint path into `configs/ant-dir-safe.json` under:
  `meta_params[encoder_type + "_path"]` (e.g. `mlp_attn_path`).

---

## Stage 4 — Train the meta-policy (`COSTA/run_meta_cpq.py` → `MFMetaSafePolicyTrainer`)
This is the main “offline RL with meta-conditioning” stage.

### 4A) Entry point (`run_meta_cpq.py`)
For AntDir:
- it creates **3 env instances** and resets task id 0/1/2
- it loads offline buffers into `meta_buffer[task]`:
  `offline_data/ant-dir-safe_<task>/offline_buffer.npz`
  with `cost_bound = safety_threshold`
- it loads `vae_list[task]` from:
  `run/ant-dir-safe_<task>/vae/vae.pt` citeturn6commentary8005533592196931

Then it constructs actor/critics where the **policy input dimension is augmented**:
- `state_dim = obs_dim + context_params.encode_dim`
- (optionally `+1` if `use_state_augmentation`) citeturn6commentary8005533592196931

It loads the context encoder:
- `encoder = torch.load("run/ant-dir-safe/context_encoder/" + meta_params[encoder_type+"_path"])` citeturn6commentary8005533592196931

Then it instantiates:
- `policy = MetaCPQPolicy(...)` (SAC-like with CPQ/CQL terms + cost critic + per-task VAEs)
- `trainer = MFMetaSafePolicyTrainer(policy, encoder, env_list, meta_buffer, ...)`
- calls `trainer.train()` citeturn6commentary8005533592196931

### 4B) Where `z` comes from during training (`MFMetaSafePolicyTrainer.sample_context`)
The trainer has a `sample_context(task, train=False)` method that builds **cost-context sequences** from the offline buffer:

- If `is_focal` is True (it is, in `run_meta_cpq.py`), then for each context sample it does:
  1. `whole_sequence = buffer.random_batch(max_episode_steps)`
  2. `cost_whole_context = cat([obs, act, next_obs, cost], dim=-1)`
  3. stacks these into `cost_context` tensor of shape roughly:
     `[context_sequence_num, T, (2*obs_dim + act_dim + 1)]` citeturn8commentary2755323138019020

Then (in the unseen part of `train()`), the typical use is:
- `z = encoder(cost_context)` (or averaged across `context_sequence_num`)
- **append `z` to observations** to form augmented state:
  `aug_obs = concat([obs, z], axis=-1)`
- train policy with batches sampled from the offline buffer.

You can confirm the same `obs/action/next_obs/cost` concatenation scheme is used in **online_adaptation** too (see below), so the encoder interface is consistent. citeturn6commentary9657373169804531

---

## Stage 5 — Online adaptation trace (`COSTA/online_adaptation.py`)
This is the clearest “end-to-end” trace because it explicitly computes `z` and concatenates it to the observation before calling the policy.

### 5A) Setup (AntDir)
- builds `env_list[0..2]` and `reset_task(task)` unless OOD
- loads:
  - VAEs: `run/ant-dir-safe_<task>/vae/vae.pt`
  - offline buffers: `offline_data/ant-dir-safe_<task>/offline_buffer.npz` into `meta_buffer[task]`
  - context encoder: same path as training
  - policy weights: `policy.load_state_dict(torch.load(_config["policy_path"]))` citeturn6commentary9657373169804531

### 5B) Initial context `z` from offline buffer (bootstrap)
For each goal (iterates 2 → 0):
1. `buffer = meta_buffer[num_episodes]`
2. `whole_sequence = buffer.random_batch(300)`
3. Builds a context tensor:
   ```python
   cost_whole_context = torch.stack([
     torch.cat([obs, act, next_obs, cost], dim=-1)
   ])
   z = encoder(cost_whole_context.squeeze().unsqueeze(0)).cpu().numpy()
   ```
So the encoder input is a (batched) sequence of `[obs, act, next_obs, cost]`. citeturn6commentary9657373169804531

### 5C) Acting uses augmented observation `[obs, z]`
At each env step:
- `policy.select_action(concat([obs, z], axis=-1), deterministic=True)` citeturn6commentary9657373169804531

### 5D) Updating `z` from collected online trajectories
They store each transition as:
`con = concat([obs, action, next_obs, cost])`

Then after an episode ends, during the “random trajectories” phase, they update:
- `z = encoder(torch.stack(exp_context[goal][-t_ep_l:]).unsqueeze(0))` citeturn6commentary9657373169804531

So: **encoder is repeatedly re-run on the most recent episode context** to produce a new `z` for the next rollouts.

---

## If you want the “missing” part (deep trace)
Two places are still “partially visible” from tooling limits:
- `OfflineMetaLearner.train()` (how it forms positives/negatives, uses cost_dynamic_models, decoder loss, etc.)
- `MFMetaSafePolicyTrainer.train()` (exactly where `sample_context()` is called and how `z` is fused into batches)

If you want, I can narrow in with targeted searches like:
- `repo:ApocalypseX/COSTA "sample_context(" "encoder(" path:COSTA/policy_trainer`
- `repo:ApocalypseX/COSTA "self.encoder" "cost_context" path:COSTA/policy_trainer`

Tell me which one you care about more: **(A) offline meta-policy training loop** or **(B) offline context-encoder training loop**.

---

**Episode** and **epoch** are two different “clocks” in RL training:

## Episode
An **episode** is one rollout from an environment reset until the environment says it’s done.

In this repo, an episode ends when either:
- the env returns `done=True` (task termination), **or**
- the rollout hits the time limit `max_ep_len` (a forced cutoff / timeout)

So **`max_ep_len` = maximum number of env steps allowed in a single episode** (e.g., 300). It prevents one trajectory from running forever.

## Epoch
An **epoch** is a training/logging unit that groups a fixed amount of data collection (and then usually a policy/value update).

Here, **`steps_per_epoch` = how many environment interaction steps to collect per epoch** (e.g., 6000). After collecting those steps, the code does its “end-of-epoch” work (updates, logging, saving, etc., depending on the algorithm).

So **epochs are measured in steps**, not in episodes.

## How they differ (with your numbers)
If:
- `steps_per_epoch = 6000`
- `max_ep_len = 300`

Then:
- Each epoch contains **exactly 6000 environment steps collected** (per worker / “local” process in their loop).
- Each episode contains **up to 300 steps**, but can be shorter if `done=True` happens earlier.

That means:
- You might *often* see around `6000 / 300 = 20` episodes **if episodes always hit the 300-step cap**.
- But you will **not** always get exactly 20 episodes because:
  - episodes can end early (done before 300), increasing episode count, and/or
  - the epoch can end in the middle of an episode (trajectory gets “cut off by epoch”), meaning the last episode may be partial and not counted as a completed episode in some logs.

## One-line mental model
- **`max_ep_len` limits a single trajectory** (episode length cap).
- **`steps_per_epoch` limits how much total experience you gather before an update/logging cycle** (epoch size cap).

Not necessarily.

In this codebase, **`steps_per_epoch` means “environment interaction steps per epoch”**, not “episodes per epoch”. So an epoch ends after **6000 timesteps** (per process / “local” worker), regardless of how many episodes that took. In `run_agent.py`, the main loop is:

- `for t in range(local_steps_per_epoch):` collect one env step each iteration
- An **episode** ends when `d` (env done) **or** when `ep_len == max_ep_len` (timeout) (`terminal = d or (ep_len == max_ep_len)`)
- If the epoch ends mid-episode (`t == local_steps_per_epoch - 1`), it **cuts the trajectory** and prints a warning (“trajectory cut off by epoch … steps”). The episode is not counted as finished for logging unless `terminal` is true.

So with `steps_per_epoch = 6000` and `max_ep_len = 300`:

- **Upper bound**: if every episode always runs to the timeout `max_ep_len` (300 steps) and never ends early, then you’d get about `6000 / 300 = 20` *full-length* episodes per epoch.
- **But in practice**, episodes can end earlier (task termination), or the last episode can be **cut off by the epoch boundary**, so the number of completed episodes in an epoch is usually **not exactly 20**—it can be **more than 20** (if episodes end early) or **~19 + one partial** (if the last one is cut off).

This behavior is visible in `safety-starter-agents/safe_rl/pg/run_agent.py` where termination is defined as `d or (ep_len == max_ep_len)`, and an epoch is enforced by `range(local_steps_per_epoch)` (with cutoffs at the last step). (Your code-search results may be incomplete due to tool limits; you can view more matches in GitHub code search.)


In that file:

## `eval()`
- **Yes, `eval()` is an inbuilt (built-in) Python function.**
- Here it’s used on line 73:

  ```python
  algo = eval('safe_rl.' + algo)
  ```

  That evaluates a string like `"safe_rl.cpo"` or `"safe_rl.ppo"` and returns the corresponding object from the `safe_rl` module (typically a function).

## `algo()`
- **No, `algo()` is not a Python built-in function.**
- In `main(task, algo, goal, seed, cpu)`, `algo` starts as a **string** argument from the CLI (e.g., `"cpo"`).
- Then the script converts it into a **callable** by doing:

  ```python
  algo = eval('safe_rl.' + algo)
  ```

- After that, when you see:

  ```python
  algo(env_fn=None, ...)
  ```

  `algo(...)` is simply “call the function (or callable) stored in the variable `algo`”, which came from `safe_rl` (e.g., `safe_rl.cpo`, `safe_rl.ppo`, etc.).

If you tell me which algorithm you ran (e.g., `--algo cpo`), I can help you locate the exact definition in the `safe_rl` package (whether it’s in this repo or an external dependency).
`algo` is callable **because the code changes what `algo` refers to**.

### 1) Before `eval`: `algo` is just a string
`main(task, algo, goal, seed, cpu)` receives `algo` from the CLI:

- default: `--algo cpo`
- so at first: `algo == "cpo"` (a **string**) citecall_8TTikEbTfOEF9S1JvdrBuKPp

A string is **not** callable.

### 2) Line 73 turns that string into a function (or other callable)
This line:

```py
algo = eval('safe_rl.' + algo)
```

If `algo` is `"cpo"`, the string becomes `"safe_rl.cpo"`, and `eval(...)` evaluates it and returns the **actual object** `safe_rl.cpo` from the imported `safe_rl` module. citecall_8TTikEbTfOEF9S1JvdrBuKPp

That returned object is expected to be a **function** (or some callable), so now **`algo` becomes callable**.

### 3) Line 106 calls that callable — it does *not* call `eval` again
Later the script does:

```py
algo(env_fn=None, ac_kwargs=..., epochs=..., env=..., ...)
```

This **calls the object currently stored in `algo`**. At this point, `algo` is no longer the string `"cpo"`; it’s the function-like object that `eval` returned earlier. citecall_8TTikEbTfOEF9S1JvdrBuKPp

So:

- `eval(...)` runs **once** at line 73 (unless the code reaches line 73 multiple times, e.g., calling `main()` multiple times).
- `algo(...)` at line 106 **does not automatically call `eval`**. It just calls the function/object that was already looked up.

### Tiny mental model
Think of it like:

```py
name = "cpo"                 # string
fn = safe_rl.cpo             # function
fn(...)                      # call function
```

Your code just does the “string → function” step using `eval`.

If you want, I can show a safer alternative to `eval` here (using `getattr(safe_rl, algo)`), which avoids evaluating arbitrary strings.





 
 
That command runs **`COSTA/online_adaptation.py`** in **“unseen task generalization / OOD (out-of-distribution)”** mode for the **CheetahVel**-style task suite, using a fixed random seed.

## Command
```bash
python online_adaptation.py --task cheetah-vel-safe --ood 1 --seed 0
```

## What each flag means (based on `online_adaptation.py`)
### `--task cheetah-vel-safe`
- Selects which **JSON config** to load:
  - The script reads `configs/<task>.json`, so here it loads:
    - `configs/cheetah-vel-safe.json`
- That config controls things like:
  - which env (`env_name`) to build,
  - how many tasks there are (`num_tasks`),
  - rollout/eval counts (`exp_trj_num`, `rnd_trj_num`, `eval_episodes`, etc.),
  - state/action dims, safety thresholds, buffer sizes, encoder settings, etc.

### `--ood 1`
- Turns on OOD task evaluation.
- In the environment construction loop, the key effect is:
  - if `ood` is true, it calls `env.reset_task(task + num_tasks)`
  - otherwise it calls `env.reset_task(task)`
- So **OOD=1 shifts the task IDs** to a disjoint set (e.g., training tasks might be `0..num_tasks-1`, and OOD tasks become `num_tasks..2*num_tasks-1`).
- This is the mechanism that makes it an **“unseen task generalization”** experiment.

### `--seed 0`
- Sets the run’s random seed (reproducibility).
- The script overwrites the config’s `"seed"` with `0` (because it only overwrites when `seed >= 0`), and then seeds:
  - PyTorch / NumPy via `seed_torch(seed)`
  - each env instance via `env.seed(seed)`

## What the script actually does when you run it
At a high level, it performs **online adaptation and evaluation** across multiple tasks:

1. **Creates `num_tasks` environment instances** using `env_name` and `env_params` from the config.
2. For each task index `t`, it:
   - switches the env to the task:
     - **OOD**: task id = `t + num_tasks`
     - **in-distribution**: task id = `t`
   - loads a **task-specific VAE** from:
     - `run/<env_name>_<t>/vae/vae.pt`
   - loads an **offline replay buffer** from:
     - `offline_data/<env_name>_<t>/offline_buffer.npz`
3. Loads the **context encoder** from a path under:
   - `run/<env_name>/context_encoder/...`
4. Builds the policy (`MetaCPQPolicy`) and loads pretrained weights from `--policy_path` (or its default if you don’t pass it).
5. Runs rollouts to build “context” (trajectories), encodes that context into a latent `z`, and then **evaluates the policy conditioned on `[obs, z]`** on each task.
6. Logs results into a run directory (the directory name includes “task_generalization” when `--ood 1`), and saves plots + `adaptation_context.npy`.

## One practical note
The command assumes you already have the required artifacts in place (otherwise you’ll hit missing-file errors), especially:
- `configs/cheetah-vel-safe.json`
- the pretrained policy file at `--policy_path` (or the script’s default path)
- `run/<env_name>_<taskid>/vae/vae.pt`
- `offline_data/<env_name>_<taskid>/offline_buffer.npz`
- the context encoder under `run/<env_name>/context_encoder/...`

If you tell me what error you get (or paste `configs/cheetah-vel-safe.json`), I can map the expected folder/file names for *your* setup exactly.
---

You **can run `run_vae.py` and `run_discriminator.py` in parallel**—there’s **no direct dependency** where one script reads the other’s saved outputs.

### Why (based on the code)
- **`run_vae.py`**:
  - Loads the offline dataset from `.../offline_buffer.npz` via `buffer.init_buffer(_config["data_path"])`.
  - Trains a `CVAE` and writes logs/models under its own timestamped output folder: `.../<task>_<goal>/vae/<timestamp>/...`
- **`run_discriminator.py`**:
  - Also loads the **same kind of offline dataset** from `.../offline_buffer.npz`.
  - Trains an `EnsembleDiscriminatorModel` and writes logs/models under its own timestamped output folder: `.../<task>_<goal>/discriminator/<timestamp>/...`

So they are **independent training jobs** that both consume the offline buffer, but **neither script loads the other’s model checkpoints**.

### Only practical constraint
Both scripts hardcode:
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
```
So if you run them at the same time on the same machine, they’ll both try to use **GPU #1**, which can cause **OOM / slowdown**. If you want true parallelism, run them on different GPUs (or edit that env var per process).

If you want, I can also check where the **trained VAE or discriminator is actually used** later (e.g., in the main algorithm training script) to tell you the intended overall sequence of the whole pipeline.

You can execute run_vae.py and run_discriminator.py in parallel; they do not depend on each other’s outputs.

Here’s why:

- Both scripts load their own data from the offline buffer (see the use of buffer.init_buffer and data_path in each).
- run_vae.py trains a CVAE model using actions and observations.
- run_discriminator.py trains a cost/discriminator model using actions, observations, next_observations, and costs.
- Neither script loads or uses the model or output produced by the other.
- The README also shows both scripts being run in parallel in the example shell scripts.

So, you do not need to run them in sequence—there is no output dependency between them.

---

In this repo, the discriminator used in `COSTA/train/discriminator_train.py` is **an ensemble MLP discriminator**: `EnsembleDiscriminatorModel` (defined in `COSTA/modules/discriminator_module.py`).

It takes the concatenated input **[state, action, next_state]** (dimension `2*obs_dim + action_dim`) and passes it through:
- a stack of **EnsembleLinear** layers with **Swish** activation (an MLP “backbone”),
- then a final **EnsembleLinear → Sigmoid** to output a **binary probability** (safe/unsafe).

`run_discriminator.py` shows it’s instantiated like:
- `model = EnsembleDiscriminatorModel(state_dim, action_dim, hidden_dims, num_ensemble=5, num_elites=5, ...)`

There’s also a simpler non-ensemble `Discriminator` class in `COSTA/nets/discriminator.py` (plain `Linear + ReLU (+ optional Dropout) + Sigmoid`), but it’s commented out in `run_discriminator.py`, so the code is currently using the **ensemble discriminator**.

---

User: does this allow to run only one GPU even if I have 4

GitHub Copilot: Yes, the line:

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

restricts the visible GPUs to only the one with index 1. This means that, even if your system has 4 GPUs, only the GPU with index 1 will be accessible to your Python process. All other GPUs (0, 2, 3) will be invisible to CUDA and PyTorch in this script.

User: how to change this to allow it use all GPUs

GitHub Copilot: To allow your script to use all available GPUs, you can either:

1. Remove or comment out the line:
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

2. Or, set it to an empty string or a comma-separated list of all GPU indices (for 4 GPUs: 0,1,2,3):
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

If you remove the line entirely, PyTorch will see all GPUs by default. If you want to be explicit, use the second option.

---

mean_eval_acc: fraction of correct predictions overall
mean_eval_acc_safe: fraction correct among safe examples (expects pred < 0.5)
mean_eval_acc_unsafe: fraction correct among unsafe examples (expects pred >= 0.5)

---


