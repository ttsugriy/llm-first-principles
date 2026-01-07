# Stage 1: Common Mistakes

## Mistake 1: Confusing Joint and Conditional Probability

**Wrong thinking**: "P(the, cat) is the probability of seeing 'the cat'"

**Correct thinking**: P(the, cat) is the joint probability of two events. For language modeling, we want P(cat | the) — the probability of "cat" *given* we've seen "the".

**The fix**:
```python
# Wrong: treating joint as conditional
p = count("the cat") / total_bigrams

# Correct: conditional probability
p = count("the cat") / count("the *")
```

---

## Mistake 2: Zero Probability = Model Broken

**Symptom**: Model assigns P=0 to valid words, causing log(0) = -inf

**Example**:
```python
>>> model.probability("elephant", context="the")
0.0  # Never seen "the elephant" in training

>>> math.log(0.0)
-inf  # Perplexity calculation breaks!
```

**The fix**: Always use smoothing
```python
# Add-one (Laplace) smoothing
p = (count + 1) / (total + vocab_size)
```

---

## Mistake 3: Not Handling Unknown Words

**Symptom**: KeyError when encountering words not in vocabulary

```python
>>> model.probability("cryptocurrency", context="about")
KeyError: 'cryptocurrency'
```

**The fix**: Add an `<UNK>` token
```python
def probability(self, word, context):
    if word not in self.vocab:
        word = "<UNK>"
    if context not in self.vocab:
        context = "<UNK>"
    # ... rest of calculation
```

---

## Mistake 4: Temperature of 0

**Wrong code**:
```python
def sample_with_temperature(probs, temperature):
    scaled = probs ** (1 / temperature)  # Division by zero when T=0!
    return scaled / sum(scaled)
```

**The fix**: Handle T=0 as argmax
```python
def sample_with_temperature(probs, temperature):
    if temperature == 0:
        return np.argmax(probs)
    scaled = probs ** (1 / temperature)
    return scaled / sum(scaled)
```

---

## Mistake 5: Log Probability Underflow

**Symptom**: Probability of long sequences becomes 0

```python
# Product of many small probabilities
p = 0.1 * 0.1 * 0.1 * ... * 0.1  # 100 terms
p = 1e-100  # Underflows to 0!
```

**The fix**: Work in log space
```python
# Sum of log probabilities
log_p = sum(math.log(p) for p in probabilities)
# Convert back only when needed
p = math.exp(log_p)
```

---

## Mistake 6: Not Shuffling Data

**Symptom**: Model only learns patterns from the beginning of text

**Wrong**:
```python
# Training on text in order
for i in range(len(text) - 1):
    train_on(text[i], text[i+1])
```

**The fix**: Shuffle training examples (for SGD-based training)

---

## Mistake 7: Perplexity on Training Data

**Symptom**: Amazing perplexity that doesn't generalize

**Wrong thinking**: "My model has perplexity 5 on training data!"

**Correct thinking**: Perplexity on training data is meaningless. Always evaluate on held-out test data.

```python
# Correct evaluation
train_text, test_text = split_data(corpus, ratio=0.9)
model.train(train_text)
perplexity = evaluate(model, test_text)  # Use TEST data
```

---

## Mistake 8: Context Boundary Handling

**Symptom**: Predictions at start of text are wrong

**Example**: For bigram model, what's P(first_word | ???)

**The fix**: Add special start/end tokens
```python
text = "<BOS> " + text + " <EOS>"
# Now P(first_word | <BOS>) is well-defined
```

---

## Mistake 9: Case Sensitivity

**Symptom**: "The" and "the" treated as different words

This fragments your counts and makes the model worse.

**The fix** (usually): Lowercase everything
```python
text = text.lower()
```

But be aware this loses information ("US" vs "us").

---

## Mistake 10: Not Normalizing Probabilities

**Symptom**: Probabilities don't sum to 1

```python
>>> sum(model.probability(w, "the") for w in vocab)
0.847  # Should be 1.0!
```

**Common cause**: Off-by-one errors in counting

**The fix**: Verify normalization
```python
def verify_distribution(model, context, vocab):
    total = sum(model.probability(w, context) for w in vocab)
    assert abs(total - 1.0) < 1e-6, f"Probs sum to {total}, not 1.0"
```
