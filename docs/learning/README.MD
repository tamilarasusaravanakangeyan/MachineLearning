# Machine Learning — Key Terms

A quick glossary of the core terms used across these learning notes, with pointers to the detailed pages.

## Types of machine learning

| Term | Definition | Learn more |
|---|---|---|
| **Supervised learning** | Training a model on data that includes the correct answers (labels), so it can predict answers for new data | [Supervised Learning](supervisedlearning/README.MD) |
| **Unsupervised learning** | Training a model on unlabeled data so it discovers hidden patterns and groupings on its own | [Unsupervised Learning](unsupervisedlearning/README.MD) |
| **Reinforcement learning** | Training an agent through rewards and penalties as it acts in an environment, producing a policy | [Reinforcement Learning](reinforcementlearning/README.MD) |
| **Generative AI** | Models that create new content (text, images, audio, video, code) from user input | [Generative AI](GenerativeAI/README.MD) |

## Core terms

| Term | Definition | Example |
|---|---|---|
| **Model** | The mathematical system that learns patterns from data and makes predictions | A spam filter |
| **Training** | The process of showing data to a model so it can learn the patterns in it | Feeding 1M labeled emails to the spam filter |
| **Prediction** | The output a trained model produces for a new, unseen input | "This email is 92% likely spam" |
| **Regression model** | A supervised model that predicts a **numeric value** | Predicting a house price: $425,000 |
| **Classification model** | A supervised model that predicts **which category** something belongs to | `spam` / `not spam` |
| **Clustering** | An unsupervised technique that groups similar data together — without predefined categories | Grouping customers into natural segments |
| **Reward** | The feedback signal (positive or negative) that guides a reinforcement learning agent | +100 for reaching the maze exit |
| **Policy** | The strategy a reinforcement learning agent learns — mapping each situation to the best action | "When I hear 'sit' → sit" |

## How the terms fit together

```
                        Machine Learning
                              │
     ┌──────────────┬─────────┴──────┬──────────────────┐
     ▼              ▼                ▼                  ▼
 Supervised     Unsupervised    Reinforcement      Generative AI
 (labels)       (no labels)     (rewards)          (creates content)
     │              │                │
 ┌───┴────┐         ▼                ▼
 ▼        ▼     Clustering        Policy
Regression Classification       (via rewards)
(number)   (category)
```

> **Tip:** Every approach shares the same core loop — a **model** goes through **training** on data and then produces **predictions** — they differ only in *what signal* teaches the model: labels, structure, or rewards.
