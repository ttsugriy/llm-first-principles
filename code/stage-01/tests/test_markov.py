"""
Test Suite for Markov Chain Language Model

These tests verify the correctness of the Markov chain implementation
and serve as executable documentation of the expected behavior.

Run with: python -m pytest tests/test_markov.py
Or simply: python tests/test_markov.py
"""

import sys
import math
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from markov import MarkovChain, SmoothedMarkovChain
from evaluate import compute_perplexity, compute_cross_entropy, compute_log_probability
from generate import generate, apply_temperature


class TestMarkovChain:
    """Tests for the basic MarkovChain class."""

    def test_bigram_counts(self):
        """Verify that bigram counting matches MLE formula."""
        model = MarkovChain(order=1)
        model.train(list("abab"))

        # From Section 1.3: P(b|a) = count(a,b) / count(a,·)
        # "abab" has bigrams: (START,a), (a,b), (b,a), (a,b), (b,END)
        # count(a,b) = 2, count(a,·) = 2 → P(b|a) = 1.0
        assert model.probability(("a",), "b") == 1.0

        # count(b,a) = 1, count(b,·) = 2 → P(a|b) = 0.5
        assert model.probability(("b",), "a") == 0.5

        # count(b,END) = 1, count(b,·) = 2 → P(END|b) = 0.5
        assert model.probability(("b",), "<END>") == 0.5

    def test_trigram_counts(self):
        """Verify trigram counting."""
        model = MarkovChain(order=2)
        model.train(list("abcabc"))

        # Context (a,b) always followed by c
        assert model.probability(("a", "b"), "c") == 1.0

        # Context (b,c) followed by a or END
        assert model.probability(("b", "c"), "a") == 0.5
        assert model.probability(("b", "c"), "<END>") == 0.5

    def test_unseen_context_returns_zero(self):
        """Unseen contexts should return probability 0."""
        model = MarkovChain(order=1)
        model.train(list("ab"))

        # 'x' was never seen in training
        assert model.probability(("x",), "a") == 0.0
        assert model.probability(("x",), "y") == 0.0

    def test_unseen_transition_returns_zero(self):
        """Unseen transitions from known contexts should return 0."""
        model = MarkovChain(order=1)
        model.train(list("ab"))

        # 'a' was seen, but 'a' → 'x' was not
        assert model.probability(("a",), "x") == 0.0

    def test_distribution_sums_to_one(self):
        """Probability distributions must sum to 1."""
        model = MarkovChain(order=1)
        model.train(list("the cat sat on the mat"))

        for history in model.counts.keys():
            dist = model.get_distribution(history)
            total = sum(dist.values())
            assert abs(total - 1.0) < 1e-10, f"Distribution for {history} sums to {total}"

    def test_order_validation(self):
        """Order must be at least 1."""
        try:
            MarkovChain(order=0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

        try:
            MarkovChain(order=-1)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_empty_training(self):
        """Training on empty data should produce minimal model."""
        model = MarkovChain(order=1)
        model.train([])
        # Empty training still has START -> END transition
        assert model.num_states() == 1  # Just the START state


class TestSmoothedMarkovChain:
    """Tests for Laplace-smoothed Markov chain."""

    def test_smoothing_prevents_zero(self):
        """Smoothing should prevent zero probabilities."""
        model = SmoothedMarkovChain(order=1, alpha=1.0)
        model.train(list("ab"))

        # Even unseen transitions should have non-zero probability
        # But only for tokens in vocabulary
        # Note: 'x' is not in vocab, so we test with vocab tokens
        prob_a_to_b = model.probability(("a",), "b")
        prob_a_to_end = model.probability(("a",), "<END>")

        assert prob_a_to_b > 0
        assert prob_a_to_end > 0

    def test_smoothing_formula(self):
        """Verify Laplace smoothing formula: (count + α) / (total + α|V|)"""
        model = SmoothedMarkovChain(order=1, alpha=1.0)
        model.train(list("aab"))  # a→a, a→b, b→END

        # count(a,a) = 1, count(a,·) = 2, |V| = 3 (a, b, END)
        # P(a|a) = (1 + 1) / (2 + 1*3) = 2/5 = 0.4
        expected = 2 / 5
        actual = model.probability(("a",), "a")
        assert abs(actual - expected) < 1e-10

    def test_smoothed_distribution_sums_to_one(self):
        """Smoothed distributions must also sum to 1."""
        model = SmoothedMarkovChain(order=1, alpha=1.0)
        model.train(list("the cat sat"))

        for history in model.counts.keys():
            dist = model.get_distribution(history)
            total = sum(dist.values())
            assert abs(total - 1.0) < 1e-10


class TestPerplexity:
    """Tests for perplexity computation."""

    def test_perfect_prediction_perplexity_one(self):
        """Perfect prediction should give perplexity close to 1."""
        model = MarkovChain(order=1)
        model.train(list("aaaa"))

        # Model learns: a always follows a, START always followed by a
        # On test "aa": P(a|START)=1, P(a|a)=1, P(END|a)=1
        # Wait, that's not quite right - let's check
        # Training "aaaa" gives: START→a, a→a, a→a, a→a, a→END
        # So P(END|a) = 1/4 = 0.25, not 1
        # Perfect prediction means the model assigns probability 1 to each token

        # Better test: train and test on same simple pattern
        model2 = MarkovChain(order=1)
        model2.train(list("a"))  # Only: START→a, a→END

        ppl = compute_perplexity(model2, list("a"))
        # P(a|START) = 1, P(END|a) = 1
        # Cross-entropy = -1/2 * (log(1) + log(1)) = 0
        # Perplexity = exp(0) = 1
        assert abs(ppl - 1.0) < 1e-10

    def test_zero_probability_gives_infinite_perplexity(self):
        """Unseen tokens should give infinite perplexity."""
        model = MarkovChain(order=1)
        model.train(list("ab"))

        # Test on "xy" - neither x nor y in training
        ppl = compute_perplexity(model, list("xy"))
        assert ppl == float('inf')

    def test_perplexity_decreases_with_order_on_training(self):
        """Higher order should fit training data better."""
        text = list("to be or not to be that is the question")

        ppls = []
        for order in range(1, 5):
            model = MarkovChain(order=order)
            model.train(text)
            ppl = compute_perplexity(model, text)
            ppls.append(ppl)

        # Each order should have lower or equal perplexity on training
        for i in range(len(ppls) - 1):
            assert ppls[i+1] <= ppls[i] + 0.01  # Allow small numerical error


class TestGeneration:
    """Tests for text generation."""

    def test_deterministic_generation(self):
        """When only one option exists, generation is deterministic."""
        model = MarkovChain(order=1)
        model.train(list("abc"))

        # a→b is the only option, b→c is the only option
        # So starting from 'a', we should always get 'abc'
        # Actually, need to think about this more carefully
        # Training: START→a, a→b, b→c, c→END
        # Generate from empty: START context, get 'a'
        # Then 'a' context, get 'b'
        # Then 'b' context, get 'c'
        # Then 'c' context, get END
        # So output should be "abc"

        from generate import generate
        result = generate(model, max_length=10)
        assert result == "abc"

    def test_temperature_extremes(self):
        """Test temperature at extremes."""
        dist = {'a': 0.7, 'b': 0.2, 'c': 0.1}

        # Low temperature: should sharpen distribution
        low_temp = apply_temperature(dist, 0.1)
        assert low_temp['a'] > 0.99  # Almost all probability on 'a'

        # High temperature: should flatten distribution
        high_temp = apply_temperature(dist, 10.0)
        # All probabilities should be closer to 1/3
        for p in high_temp.values():
            assert 0.2 < p < 0.5

    def test_temperature_one_unchanged(self):
        """Temperature 1.0 should leave distribution unchanged."""
        dist = {'a': 0.7, 'b': 0.2, 'c': 0.1}
        result = apply_temperature(dist, 1.0)

        for k in dist:
            assert abs(dist[k] - result[k]) < 1e-10


class TestMathematicalProperties:
    """Tests verifying key mathematical claims from the text."""

    def test_mle_equals_counting(self):
        """
        Verify claim from Section 1.3: MLE solution equals count ratios.

        This is the core mathematical result of Stage 1.
        """
        model = MarkovChain(order=1)
        text = list("abracadabra")
        model.train(text)

        # Manually count transitions
        from collections import Counter
        bigrams = Counter()
        context_counts = Counter()
        padded = ['<START>'] + text + ['<END>']

        for i in range(len(padded) - 1):
            ctx = padded[i]
            nxt = padded[i + 1]
            bigrams[(ctx, nxt)] += 1
            context_counts[ctx] += 1

        # Verify model probabilities match count ratios
        for (ctx, nxt), count in bigrams.items():
            expected = count / context_counts[ctx]
            actual = model.probability((ctx,), nxt)
            assert abs(expected - actual) < 1e-10, \
                f"P({nxt}|{ctx}): expected {expected}, got {actual}"

    def test_chain_rule_decomposition(self):
        """
        Verify chain rule: P(x1...xn) = prod P(xi | x1...x_{i-1})

        For Markov models, this becomes:
        P(x1...xn) = prod P(xi | x_{i-k}...x_{i-1})
        """
        model = MarkovChain(order=1)
        text = list("ab")
        model.train(text)

        # P("ab") = P(a|START) * P(b|a) * P(END|b)
        log_prob, n = compute_log_probability(model, text)

        # Compute manually
        p_a_given_start = model.probability(("<START>",), "a")
        p_b_given_a = model.probability(("a",), "b")
        p_end_given_b = model.probability(("b",), "<END>")

        expected_log = (
            math.log(p_a_given_start) +
            math.log(p_b_given_a) +
            math.log(p_end_given_b)
        )

        assert abs(log_prob - expected_log) < 1e-10

    def test_cross_entropy_perplexity_relationship(self):
        """
        Verify: perplexity = exp(cross_entropy)

        From Section 1.5.
        """
        model = SmoothedMarkovChain(order=1, alpha=0.1)
        text = list("the quick brown fox")
        model.train(text)

        ce = compute_cross_entropy(model, text)
        ppl = compute_perplexity(model, text)

        assert abs(ppl - math.exp(ce)) < 1e-10


def run_tests():
    """Run all tests and report results."""
    import traceback

    test_classes = [
        TestMarkovChain,
        TestSmoothedMarkovChain,
        TestPerplexity,
        TestGeneration,
        TestMathematicalProperties,
    ]

    total = 0
    passed = 0
    failed = []

    for cls in test_classes:
        instance = cls()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                total += 1
                try:
                    getattr(instance, method_name)()
                    passed += 1
                    print(f"  PASS: {cls.__name__}.{method_name}")
                except Exception as e:
                    failed.append((cls.__name__, method_name, e))
                    print(f"  FAIL: {cls.__name__}.{method_name}")
                    traceback.print_exc()

    print()
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if failed:
        print("\nFailed tests:")
        for cls_name, method, error in failed:
            print(f"  - {cls_name}.{method}: {error}")
        return 1
    else:
        print("\nAll tests passed!")
        return 0


if __name__ == "__main__":
    exit(run_tests())
