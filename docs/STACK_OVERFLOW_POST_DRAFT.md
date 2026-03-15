# Stack Overflow Post Draft

Stack Overflow 不是專案託管平台，所以不建議把整個作品當成「展示文」直接發文。  
比較安全的做法，是發一篇聚焦在單一技術問題的問答，並在內容中附上最小可重現程式碼。

下面是一份較符合 Stack Overflow 風格的自問自答草稿。

## Title

How can I switch between a DQN policy and tabular Q-learning in the same Python dialogue runtime?

## Question

I am building a Python dialogue runtime for an elderly-care companion prototype.

The system has:

- 12 scripted dialogue actions
- a 5-dimensional state vector
- one runtime that should support both:
  - a DQN-based policy
  - a tabular Q-learning policy

I want the runtime code to stay the same while only swapping the policy implementation.

My current approach is:

```python
if algorithm == "dqn":
    agent = DuelingDQNAgent(state_dim=5, action_dim=12)
else:
    agent = TabularQLearningAgent(state_dim=5, action_dim=12)

predicted_action = agent.select_action(state, training=False)
action_idx = int(np.argmax(predicted_action))
```

Both agents expose the same interface:

- `select_action(state, training=False)`
- `get_q_values(state)`
- `save_model(path)`
- `load_model(path)`

The runtime also filters candidate actions by slot completion, so I only want to rank a subset of the 12 actions instead of always taking the global argmax.

What is a clean way to keep one shared runtime while:

1. using the same API for DQN and Q-learning
2. letting the runtime rank only allowed actions
3. keeping training code separate from inference code

## Self-answer

The cleanest approach was to standardize the policy interface first and let the runtime operate only on that abstraction.

I used these rules:

1. Both agents return a one-hot action from `select_action(...)`.
2. Both agents expose raw action scores through `get_q_values(state)`.
3. The runtime never decides *how* Q-values are produced. It only:
   - builds the state vector
   - computes the allowed action IDs
   - ranks those action IDs with the policy scores

Example:

```python
def choose_script_id(agent, state, allowed_script_ids):
    q_values = agent.get_q_values(state)
    ranked = sorted(
        allowed_script_ids,
        key=lambda script_id: float(q_values[script_id - 1]),
        reverse=True,
    )
    return ranked[0]
```

That way:

- DQN and Q-learning stay swappable
- the runtime remains policy-agnostic
- training logic can stay in a separate trainer module

In my project, this ended up as:

- `integrated_dqn_train.py` for offline training
- `virtual_child_rl_system.py` for runtime inference

The important part was not “which RL algorithm is better”, but making both implementations obey the same runtime contract.
