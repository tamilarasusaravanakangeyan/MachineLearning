# Supervised Learning

Supervised learning models can make predictions after seeing lots of data with the **correct answers** and then discovering the connections between the elements in the data that produce the correct answers.

> **Analogy:** This is like a student learning new material by studying old exams that contain both questions *and* answers. Once the student has trained on enough old exams, the student is well prepared to take a new exam.

These ML systems are **"supervised"** in the sense that a human gives the ML system data with the known correct results (called **labels**).

```
Training data (features + labels)  ──►  Model learns patterns  ──►  Predictions on new, unseen data
```

The two most common use cases for supervised learning are **regression** and **classification**.

---

## 1. Regression

A regression model predicts a **numeric value** (a continuous quantity).

**Example:** A weather model that predicts the amount of rain, in inches or millimeters, is a regression model.

### More examples of regression models

| Scenario | Possible input data (features) | Numeric prediction (label) |
|---|---|---|
| **Future house price** | Square footage, zip code, number of bedrooms and bathrooms, lot size, mortgage interest rate, property tax rate, construction costs, number of homes for sale in the area | The price of the home (e.g., **$425,000**) |
| **Future ride time** | Historical traffic conditions (from smartphones, traffic sensors, ride-hailing apps), distance from destination, weather conditions | Time to arrive at the destination (e.g., **23 min 30 s**) |
| **Student exam score** | Hours studied, attendance rate, previous grades | Score out of 100 (e.g., **87.5**) |
| **Electricity demand** | Time of day, temperature, day of week, holiday or not | Power demand in megawatts (e.g., **1,240 MW**) |

### Quick example (intuition)

Suppose we have house sales data:

| Square footage | Bedrooms | Sold price |
|---|---|---|
| 1,000 | 2 | $200,000 |
| 1,500 | 3 | $300,000 |
| 2,000 | 4 | $400,000 |

The model learns the pattern (roughly *price ≈ $200 per sq ft*) and can now predict that a **1,750 sq ft** house will sell for about **$350,000** — a number it never saw during training.

### Common regression algorithms

- [Linear Regression](linear-regression.md) — deep dive on SSE, R², statistical significance, and prediction intervals
- Polynomial Regression
- Decision Tree / Random Forest Regressors
- Gradient Boosting (XGBoost, LightGBM)

---

## 2. Classification

Classification models predict the **likelihood that something belongs to a category**.

Unlike regression models, whose output is a number, classification models output a value that states **whether or not something belongs to a particular category**.

**Examples:**
- Is this email **spam** or **not spam**?
- Does this photo **contain a cat** or **not**?

Classification models are divided into two groups:

### 2.1 Binary classification

The model outputs a value from a class that contains **only two values**.

| Scenario | Possible input data | Prediction (one of two classes) |
|---|---|---|
| **Spam detection** | Email text, sender address, number of links | `spam` / `not spam` |
| **Rain forecast** | Humidity, pressure, temperature | `rain` / `no rain` |
| **Loan approval** | Income, credit score, existing debt | `approve` / `reject` |
| **Disease screening** | Test results, symptoms, age | `positive` / `negative` |

### 2.2 Multiclass classification

The model outputs a value from a class that contains **more than two values**.

| Scenario | Possible input data | Prediction (one of many classes) |
|---|---|---|
| **Precipitation type** | Temperature, humidity, pressure | `rain` / `hail` / `snow` / `sleet` |
| **Handwritten digit recognition** | Pixel values of an image | `0`–`9` (10 classes) |
| **Animal in a photo** | Image pixels | `cat` / `dog` / `bird` / `horse` |
| **News topic** | Article text | `sports` / `politics` / `tech` / `business` |

### Quick example (intuition)

A spam filter is trained on thousands of emails already labeled by humans:

```
"WIN A FREE PRIZE!!! Click here"   → spam
"Meeting moved to 3pm tomorrow"    → not spam
"Cheap pills, limited offer!!!"    → spam
"Your invoice for August"          → not spam
```

The model learns which words and patterns tend to appear in spam. When a new email arrives, it outputs a probability, e.g. **"92% likely spam"**, and the email is routed to the spam folder.

### Common classification algorithms

- Logistic Regression
- Decision Trees / Random Forests
- Support Vector Machines (SVM)
- Naive Bayes
- Neural Networks

---

## Regression vs. Classification at a glance

| | Regression | Classification |
|---|---|---|
| **Output** | Continuous number | Category / class |
| **Example question** | "How much rain will fall?" | "Will it rain — yes or no?" |
| **Example output** | `2.3 inches` | `rain` |
| **Typical metrics** | MAE, MSE, RMSE, R² | Accuracy, Precision, Recall, F1 |

> **Tip:** The same real-world problem can often be framed either way. *"How many millimeters of rain tomorrow?"* is regression; *"Will it rain tomorrow?"* is binary classification.
