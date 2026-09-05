# Generative AI

Generative AI is a class of models that **creates content from user input**. For example, generative AI can create unique images, music compositions, and jokes; it can summarize articles, explain how to perform a task, or edit a photo.

```
User input (prompt)  ──►  Generative model  ──►  New content (text / image / audio / video / code)
```

Generative AI can take a variety of inputs and create a variety of outputs — **text, images, audio, and video** — and also **combinations** of these. For example, a model can take an image as input and create an image *and* text as output, or take an image *and* text as input and create a video as output.

---

## 1. Input-to-output naming

We can describe generative models by their inputs and outputs, typically written as **"type of input"-to-"type of output"**. A partial list:

- **Text-to-text** — chatbots, summarizers, translators
- **Text-to-image** — image generators (Imagen, DALL·E, Midjourney)
- **Text-to-video** — video generators (Phenaki, Sora)
- **Text-to-code** — coding assistants (Copilot, Claude Code)
- **Text-to-speech** — voice synthesis
- **Image-and-text-to-image** — photo editing with instructions

### Examples of generative models

| Model type | Example input | Example output |
|---|---|---|
| **Text-to-text** | *"Who invented the Le Mans race?"* | A paragraph explaining that the 24 Hours of Le Mans was founded by the Automobile Club de l'Ouest, with the first race held in May 1923 near Le Mans, France. *(Source: Bard)* |
| **Text-to-image** | *"An alien octopus floats through a portal while reading a newspaper."* | Image of a cartoon octopus. *(Source: Imagen)* |
| **Text-to-video** | *"A photorealistic teddy bear is swimming in the ocean at San Francisco. The teddy bear goes under water… A panda bear is swimming under water."* | Video of a teddy bear swimming underwater. *(Source: Phenaki)* |
| **Text-to-code** | *"Write a Python loop that loops over a list of numbers and prints the prime numbers."* | See code below. *(Source: Bard)* |
| **Image-to-text** | Image of a flamingo. | *"This is a flamingo. They are found in the Caribbean."* *(Source: Google DeepMind)* |

**Text-to-code example output:**

```python
for number in numbers:
  # Check if the number is prime.
  is_prime = True
  for i in range(2, number):
    if number % i == 0:
        is_prime = False
        break
  # If the number is prime, print it.
  if is_prime:
    print(number)
```

---

## 2. How does generative AI work?

At a high level, generative models **learn patterns in data with the goal of producing new but similar data.** Generative models are like:

- **Comedians** who learn to imitate others by observing people's behaviors and style of speaking
- **Artists** who learn to paint in a particular style by studying lots of paintings in that style
- **Cover bands** that learn to sound like a specific music group by listening to lots of music by that group

### Training pipeline

To produce unique and creative outputs, generative models are trained in stages:

```
Stage 1: Unsupervised pre-training      Stage 2: Fine-tuning (optional)
┌─────────────────────────────┐         ┌──────────────────────────────────┐
│ Learn to mimic huge amounts │   ──►   │ Supervised or reinforcement      │
│ of unlabeled data           │         │ learning on specific tasks       │
│ (books, images, code…)      │         │ (summarize articles, edit photos)│
└─────────────────────────────┘         └──────────────────────────────────┘
```

1. **Initially trained using an unsupervised approach** — the model learns to mimic the data it's trained on (e.g., predicting the next word across billions of sentences).
2. **Sometimes trained further using supervised or reinforcement learning** on specific data related to tasks the model might be asked to perform — for example, summarizing an article or editing a photo.

### Worked example (intuition)

A text model trained on millions of sentences learns that after *"The capital of France is"* the most likely next word is *"Paris."* Repeat that next-word skill over long passages and the model can write essays, answer questions, and hold conversations — all from patterns it absorbed during training.

---

## 3. Real-world use cases

Generative AI is a quickly evolving technology with new use cases constantly being discovered. For example:

| Area | Use case |
|---|---|
| **E-commerce** | Automatically removing distracting backgrounds from product images; improving quality of low-resolution images |
| **Writing** | Drafting emails, summarizing long documents, translation |
| **Software** | Generating and reviewing code, writing tests |
| **Media** | Creating concept art, music, and video from descriptions |
| **Customer support** | Chatbots that answer product questions in natural language |

> **Tip:** Generative AI differs from the other ML types by its *goal*. Supervised/unsupervised/reinforcement models **predict or discover** (a label, a cluster, an action); generative models **create** new content that never existed in the training data.
