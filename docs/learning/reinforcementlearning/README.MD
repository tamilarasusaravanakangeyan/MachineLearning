# Reinforcement Learning

Reinforcement learning (RL) models make predictions by getting **rewards or penalties** based on actions performed within an **environment**. A reinforcement learning system generates a **policy** that defines the best strategy for getting the most rewards.

> **Analogy:** Training a dog. Nobody hands the dog a labeled dataset of "correct sits." The dog tries actions — sit, bark, roll over — and gets a treat (reward) or nothing (penalty). Over time it learns a strategy: *"when I hear 'sit' and I sit, I get a treat."* That learned strategy is the **policy**.

```
        ┌────────────── Agent ──────────────┐
        │   (learns a policy: state → action)│
        └───────┬───────────────────▲───────┘
         action │                   │ reward + new state
        ┌───────▼───────────────────┴───────┐
        │            Environment            │
        └───────────────────────────────────┘
```

The key difference from supervised learning: **nobody provides the correct answers.** The model learns by **trial and error**, guided only by the reward signal.

---

## Core concepts

| Term | Meaning | Dog-training example |
|---|---|---|
| **Agent** | The learner / decision maker | The dog |
| **Environment** | The world the agent acts in | Your living room |
| **Action** | What the agent can do | Sit, bark, roll over |
| **State** | The current situation | Owner just said "sit" |
| **Reward** | Feedback signal (positive or negative) | Treat (+1), nothing (0) |
| **Policy** | The learned strategy mapping states to actions | "When I hear 'sit' → sit" |

---

## Examples of reinforcement learning

| Application | Agent | Actions | Reward |
|---|---|---|---|
| **Robot walking** | Robot | Move joints/motors | + for moving forward, − for falling |
| **AlphaGo (game of Go)** | Game-playing program | Place a stone on the board | + for winning the game, − for losing |
| **Self-driving car** | Driving software | Steer, accelerate, brake | + for smooth safe progress, − for collisions or lane violations |
| **Video game AI** | Game bot | Controller inputs | Game score |
| **Data center cooling** | Control system | Adjust fans/valves | − energy used (less is better), while staying in safe temperature range |

### Worked example: teaching an agent to play a maze

1. The agent starts with **no knowledge** — it moves randomly.
2. Reaching the exit gives **+100**; hitting a wall gives **−1**; each step costs **−0.1** (so shorter paths are better).
3. Early episodes: mostly random wandering, low scores.
4. Over thousands of episodes, the agent notices which turns led to the exit and updates its policy.
5. Eventually the policy converges: from any position, the agent knows the action that leads to the exit fastest.

No one ever told it the route — the reward signal alone shaped the behavior.

---

## How RL differs from the other approaches

| | Supervised | Unsupervised | Reinforcement |
|---|---|---|---|
| **Training signal** | Correct answers (labels) | None | Rewards / penalties |
| **Learns from** | A fixed labeled dataset | A fixed unlabeled dataset | Its own actions (trial and error) |
| **Output** | Prediction | Groups / patterns | Policy (best action per state) |
| **Example** | "Is this email spam?" | "What customer groups exist?" | "What move wins the game?" |

### Common RL algorithms

- Q-Learning / Deep Q-Networks (DQN)
- Policy Gradient methods (REINFORCE)
- Actor–Critic methods (A2C, PPO)

> **Tip:** RL shines when the "right answer" is hard to label but success is easy to measure — you can't label every correct joystick movement in a game, but you can always read the score.
