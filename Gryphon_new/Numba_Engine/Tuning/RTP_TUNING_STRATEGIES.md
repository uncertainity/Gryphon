# Low-Level Strategies for Slot RTP Tuning

## 1. Purpose

This document describes a general engineering approach for tuning slot-game math to one or more statistical targets, such as:

- Total RTP
- Base-game RTP
- Feature RTP
- Any-win and base-win hit rates
- Feature-trigger rates
- Retrigger rates
- Jackpot frequencies
- Volatility and win-distribution targets
- Maximum-win exposure

RTP tuning is not normally a single root-finding problem. A slot configuration is a mixture of continuous probabilities, integer weights, reel-strip arrangements, pay values, and dependent feature mechanics. The appropriate method depends on the parameter being changed.

The central principle is to separate the problem into measurable components and assign each target to the smallest set of controls that can influence it predictably.

---

## 2. Define Every Metric Before Tuning

A target must have an exact operational definition. For example, "24% hit rate" could mean any of the following:

- A base spin has a positive base-game award.
- A paid round has any award, including a feature award.
- At least one payline or ways win occurs.
- A feature triggers, even if its eventual award is zero.
- Each individual winning event is counted as a hit.

These are different statistics. Define all denominators and counting rules before optimization.

Typical definitions are:

```text
total_rtp       = total_paid_round_win / total_paid_round_bet
base_rtp        = total_base_win / total_paid_round_bet
feature_rtp     = total_feature_win / total_paid_round_bet
any_hit_rate    = paid_rounds_with_total_win_gt_0 / paid_rounds
base_hit_rate   = paid_rounds_with_base_win_gt_0 / paid_rounds
trigger_rate    = feature_triggers / paid_rounds
retrigger_rate  = retriggers / feature_sessions
```

Usually:

```text
total_rtp = base_rtp + feature_rtp + jackpot_rtp + other_rtp_components
```

Write targets with tolerances rather than as exact floating-point equalities:

```text
Total RTP:          0.9600 +/- 0.0010
Any-win hit rate:   0.2400 +/- 0.0020
Feature trigger:    1 / 80 +/- 3%
Jackpot trigger:    1 / 2,000,000 within an agreed confidence bound
```

---

## 3. Audit the Model Before Tuning

Never optimize a configuration until it is clear that every configured value reaches the intended runtime calculation.

Check for:

- A configured weight being ignored or replaced by another weight array
- Off-by-one symbol and paytable indices
- Incorrect wager normalization
- Feature wins counted twice or omitted
- Caps applied at the wrong stage
- Integer rounding before aggregation
- A target metric whose counter has a different definition
- Config objects that are copied shallowly and mutated unexpectedly
- Random seeds that are logged but not actually used
- Separate evaluators implementing different game rules
- Dead configuration fields

Useful audit tests include:

1. Give one outcome all the weight and verify that it always occurs.
2. Give one prize a distinctive value and verify where it appears in reported RTP.
3. Disable each feature independently and verify that its RTP contribution becomes zero.
4. Use a tiny hand-checkable reelset and enumerate every possible stop.
5. Confirm that total RTP equals the sum of all reported RTP components.

Optimization will exploit implementation defects if they are present. A mathematically excellent result from an incorrect evaluator is still an incorrect game.

---

## 4. Build a Reproducible Evaluation Harness

The evaluator should accept both a candidate configuration and an explicit seed:

```text
evaluate(config, seed, spin_count) -> metrics
```

At minimum, return:

```text
spin_count
total_bet
total_win
total_rtp
base_rtp
feature_rtp_by_type
any_win_hit_rate
base_hit_rate
trigger_counts_by_type
average_feature_award
win_variance
maximum_observed_win
win-range histogram
```

Do not hide candidate configuration in global state. This makes parallel evaluations, reproduction, and comparisons harder.

### Common random numbers

When comparing two nearby configurations, evaluate both with the same random-number streams where practical. This is called the common-random-numbers technique. It reduces the noise in the difference between candidates.

The seed plan should be explicit:

```text
generation_seed = hash(master_seed, candidate_id, "generation")
simulation_seed = hash(master_seed, candidate_id, "simulation")
validation_seed = independent seed not used during search
```

For parallel workers, pass a distinct derived seed to every task. Do not depend on inherited global RNG state.

---

## 5. Exact Evaluation Versus Simulation

Use exact calculation wherever the state space is manageable.

For independent reel stops with reel lengths \(L_1, L_2, \ldots, L_n\), the number of stop combinations is:

\[
N_{states}=\prod_{i=1}^{n}L_i
\]

An exact evaluator can produce exact base RTP, hit rate, symbol contribution, and trigger-window frequency. It avoids Monte Carlo noise and is especially valuable during local reel-order optimization.

Dynamic features with persistent boards, respins, collectors, multipliers, or jackpots may require:

- A state-transition/Markov calculation
- Dynamic programming
- Recursive expectation calculation
- Monte Carlo simulation
- A hybrid exact-plus-simulation evaluator

A common hybrid is:

```text
exact base enumeration
    + exact feature-trigger probability
    + simulated or analytically computed conditional feature EV
```

Feature RTP can then be computed from:

\[
RTP_{feature}
=P(\text{trigger})
\times E[\text{feature award}\mid\text{trigger}]
\div \text{bet}
\]

This decomposition makes feature frequency and feature value separately tunable.

---

## 6. Statistical Accuracy for Simulated Metrics

If spin returns are \(X_1,\ldots,X_N\), simulated RTP is:

\[
\widehat{RTP}=\frac{1}{N}\sum_{i=1}^{N}X_i
\]

when each \(X_i\) is already normalized by the paid-round bet.

The spin-return standard deviation is not the uncertainty of estimated RTP. The approximate standard error is:

\[
SE(\widehat{RTP})=\frac{s_X}{\sqrt{N}}
\]

An approximate 95% confidence interval is:

\[
\widehat{RTP}\pm1.96\,SE(\widehat{RTP})
\]

For a hit or trigger probability \(\hat p\):

\[
SE(\hat p)\approx\sqrt{\frac{\hat p(1-\hat p)}{N}}
\]

Rare jackpots and highly skewed features may require much larger samples, exact probability calculations, batch means, or specialized rare-event methods. A normal confidence interval should not be trusted blindly for an event observed only a few times.

Use progressive simulation budgets:

```text
Exploration:       small N, many candidates
Refinement:        medium N, surviving candidates
Final selection:   large N, few candidates
Certification:     independent seeds and the required production-scale N
```

---

## 7. Classify the Available Tuning Knobs

Group parameters by the behavior they primarily control.

### Frequency controls

- Reel symbol counts
- Scatter/bonus counts
- Feature-selection probabilities
- Trigger thresholds
- Blank or losing outcome weights
- Coin/drop probabilities

### Award-value controls

- Paytable values
- Coin-value weights
- Multiplier weights
- Jackpot values
- Collector/booster strength
- Feature starting awards

### Persistence controls

- Respins awarded
- Reset-on-hit rules
- Retrigger probabilities
- Meter progression
- Board expansion/unlock probabilities

### Shape and volatility controls

- Symbol stacking
- Reel-strip ordering
- Wild placement
- Correlation between reels
- Prize concentration
- Jackpot allocation
- Caps and award truncation

Prefer controls with limited side effects. For example, after fixing feature frequency, tune feature RTP using conditional feature value rather than changing the trigger rate again.

---

## 8. Allocate RTP Before Tuning the Total

A total target such as 0.96 does not specify the intended experience. Many incompatible games can have the same total RTP.

Create an RTP budget:

```text
Regular base wins       0.55
Base modifier           0.10
Primary feature         0.27
Jackpots                0.04
Total                   0.96
```

The actual allocation is a design decision. It should be checked together with hit rate, volatility, and feature frequency.

Once trigger probability is fixed, the required conditional feature EV follows from:

\[
E[W\mid trigger]
=\frac{RTP_{feature}\times bet}{P(trigger)}
\]

This equation is a useful feasibility test. It may reveal that the requested trigger frequency and feature RTP require an implausibly high or low average feature award.

---

## 9. Scalar Root-Finding Methods

Root-finding is appropriate when one parameter \(x\) has a reasonably smooth and monotonic relationship with one target.

Define:

\[
f(x)=metric(x)-target
\]

### Bracket discovery

A valid bracket \([a,b]\) satisfies:

\[
f(a)f(b)<0
\]

To discover one:

1. Evaluate the starting point \(x_0\).
2. Probe \(x_0-\Delta\) and/or \(x_0+\Delta\).
3. Identify the direction moving toward the target.
4. Expand the step geometrically: \(\Delta,2\Delta,4\Delta,\ldots\).
5. Stop on a sign change or at the legal parameter boundary.

If no sign change exists in the legal range, that parameter cannot reach the target under the current fixed configuration.

### Bisection

For a valid bracket, repeatedly evaluate:

\[
m=\frac{a+b}{2}
\]

Keep the half that retains the sign change. Bisection is slow but reliable for deterministic, continuous, monotonic responses.

### Secant method

Given two evaluated points, propose:

\[
x_{n+1}=x_n-f(x_n)\frac{x_n-x_{n-1}}{f(x_n)-f(x_{n-1})}
\]

Secant is often faster than bisection, but it does not preserve a bracket and can jump outside legal bounds.

### Brent's method

Brent's method combines interpolation with bracket-preserving fallbacks. It is generally a better default than raw secant when deterministic evaluations and a valid bracket are available.

### Simulation precautions

For noisy objectives:

- Reuse common random numbers.
- Require the apparent improvement to exceed expected simulation noise.
- Re-evaluate promising points with a larger sample.
- Preserve legal bounds.
- Prefer stochastic approximation or bracketed searches with uncertainty-aware decisions.

Do not declare a bracket merely because two noisy point estimates fall just below and above the target. Their confidence intervals may overlap heavily.

---

## 10. Discrete Weight Tuning

Most weight tables are integer-valued, and multiplying every weight by the same constant changes nothing. Optimize normalized probabilities or weight ratios, not raw scale.

For weights \(w_1,\ldots,w_k\):

\[
p_i=\frac{w_i}{\sum_jw_j}
\]

A useful parameterization is a softmax over unconstrained values \(z_i\):

\[
p_i=\frac{e^{z_i}}{\sum_je^{z_j}}
\]

The optimizer can work in \(z\)-space. Convert the final probabilities to bounded integer weights and then search nearby integer combinations.

For a simple two-outcome table such as `[None, Trigger]`, optimize the trigger probability directly:

\[
p_{trigger}=\frac{w_{trigger}}{w_{none}+w_{trigger}}
\]

After rounding weights:

1. Reduce them by their greatest common divisor where appropriate.
2. Verify that probabilities remain within tolerance.
3. Re-evaluate the complete game.

---

## 11. Multi-Target and Multi-Parameter Optimization

When RTP, hit rate, trigger rates, and volatility must be matched simultaneously, one-dimensional secant or bisection is insufficient.

Define normalized errors:

\[
e_i=\frac{metric_i-target_i}{tolerance_i}
\]

One scalar loss is:

\[
L=\sum_iw_ie_i^2+penalties
\]

Possible penalties include:

- Invalid or negative weights
- Trigger rates outside hard bounds
- Maximum-win violations
- Incorrect RTP allocation
- Forbidden reel patterns
- Excessive jackpot exposure
- Unacceptable volatility or dead-spin rate

Better still, retain a Pareto frontier instead of collapsing every objective into one score. A candidate is Pareto-dominated if another candidate is at least as good on every objective and better on at least one.

Useful optimizers include:

- Coordinate descent for mostly separable controls
- Grid or random search for small bounded spaces
- Bayesian optimization for expensive evaluations with few dimensions
- CMA-ES for noisy continuous parameterizations
- Simulated annealing for discrete configurations
- Genetic/evolutionary algorithms for reel strips and mixed variables
- NSGA-II or another multi-objective evolutionary method for Pareto search
- Mixed-integer optimization when an accurate analytic model is available

A practical staged approach is usually easier to control than a single unconstrained global optimization.

---

## 12. Reelset Tuning: Separate Counts From Ordering

Reel strips are difficult because they contain two different design problems.

### Symbol-count optimization

Counts determine marginal symbol frequencies:

```text
number of each low/mid/high/premium symbol per reel
number of Wilds per reel
number of Scatters or feature symbols per reel
```

Counts strongly influence RTP, hit rate, and trigger opportunity. Optimize integer counts first while honoring reel length and symbol restrictions.

Avoid regenerating counts randomly every time a count configuration is evaluated. That confounds configuration quality with generator luck. Treat the exact count matrix as part of the candidate.

### Symbol-order optimization

Ordering determines adjacency and visible-window behavior:

- Stack lengths
- Multiple instances visible in one window
- Wild adjacency
- Scatter spacing
- Near-miss behavior
- Ways/payline correlations
- Volatility

Once counts are fixed, modify ordering with count-preserving moves:

- Swap two positions
- Move one symbol to another position
- Reverse a segment
- Rotate a segment
- Split or merge a stack
- Exchange two equal-length segments
- Swap corresponding portions between parent candidates

These operations let the optimizer change reel behavior without changing symbol frequency.

### Circular constraints

A reel strip is circular. Every validation rule must inspect the boundary between the final and first positions.

Check:

- Maximum exact-symbol run
- Maximum symbol-group run
- Minimum distance between special symbols
- Forbidden neighboring symbols
- Visible-window special counts
- Runs formed by joining the end and beginning

A generator that caps each sampled chunk does not necessarily cap the final run: two consecutive chunks of the same symbol or group merge into one run.

---

## 13. Markov and Grammar-Based Reel Generation

A grammar is useful for generating plausible initial strips from a compact configuration. It should be treated as a proposal generator, followed by evaluation and local optimization.

A first-order Markov model uses:

\[
P(S_{t+1}=j\mid S_t=i)
\]

When exact symbol counts must be consumed, use a constrained transition score:

\[
P(j\mid i,remaining)
\propto remaining(j)\exp(\theta_{ij})
\]

where \(\theta_{ij}\) expresses preferences such as:

- Encourage low-symbol stacks
- Discourage premium-symbol stacks
- Prevent adjacent Wilds
- Control Scatter spacing
- Encourage or discourage transitions between symbol groups

Higher-order models can condition on the last two or more symbols, but their parameter count grows quickly.

### Required generator safeguards

- Update previous-state variables after every placement.
- Index transition weights by candidate position, not by raw symbol ID.
- Use floating-point probability arrays.
- Validate all probabilities before sampling.
- Keep parameters in their legal ranges.
- Check whether the remaining multiset can still satisfy all constraints.
- Validate the completed strip, including its circular boundary.
- Retry or backtrack when construction reaches an infeasible tail.
- Save the final strips, config, and generation seed.

### Run-based generation

A run-length distribution can complement a Markov transition model. However, sampled chunk size and actual final run size are not automatically the same. Prevent consecutive same-symbol/group chunks or merge and validate them explicitly.

### Generator-level optimization

Compact grammar parameters can themselves be optimized:

```text
group proportions
within-group symbol proportions
self-transition preferences
special-symbol separation parameters
stack-length distribution
transition matrix coefficients
```

Because one grammar configuration can generate many strips, estimate its quality using multiple generated reelsets:

\[
score(config)=E_{strip\sim generator(config)}[loss(strip)]
\]

Also track score variance. A stable generator is preferable to one that occasionally produces an excellent strip but usually produces poor ones.

---

## 14. Recommended Reel Search Pipeline

```text
1. Generate legal integer count matrices.
2. Reject obviously infeasible count matrices analytically.
3. Generate multiple circular strip arrangements for each matrix.
4. Run exact or low-cost initial evaluation.
5. Retain Pareto candidates.
6. Apply count-preserving local mutations.
7. Re-evaluate only affected metrics where incremental evaluation is possible.
8. Increase the evaluation budget for survivors.
9. Tune non-reel feature values around the surviving reelsets.
10. Validate finalists independently.
```

For local search:

```text
current = initial legal reelset
best = current

repeat:
    proposal = count_preserving_mutation(current)
    if not circular_constraints_hold(proposal):
        continue

    metrics = evaluate(proposal)
    loss = score(metrics)

    accept if improved
    or accept probabilistically under simulated annealing

    update best
```

Maintain diversity. Keeping only the single current best reelset can cause premature convergence.

---

## 15. Staged End-to-End Tuning Workflow

### Stage A: Establish feasibility

1. Audit all math and counters.
2. Evaluate the baseline configuration.
3. Determine legal parameter bounds.
4. Test extreme configurations.
5. Confirm that all requested targets are reachable together.

### Stage B: Tune occurrence rates

Tune relatively independent frequency targets first:

- Feature triggers
- Trigger-type mix
- Retriggers
- Jackpot opportunities

Do not compensate for an incorrect trigger rate solely by changing award values.

### Stage C: Tune hit rate and base behavior

Adjust reel counts and ordering, low-value wins, Wild behavior, or other frequency-oriented controls. Track base RTP at the same time.

### Stage D: Tune RTP allocation

Set base, feature, jackpot, and secondary-component RTP near their budgets. Tune conditional feature value after feature frequency is stable.

### Stage E: Tune distribution shape

Check:

- Standard deviation/volatility
- Win bands
- Zero-win frequency
- Median and percentile behavior
- Maximum exposure
- Feature duration
- Jackpot contribution

Two games with identical RTP and hit rate can still have radically different distributions.

### Stage F: Joint refinement

Cycle through parameter groups because later stages may disturb earlier metrics:

```text
trigger rates
  -> hit rate
  -> base RTP
  -> conditional feature EV
  -> total RTP
  -> distribution constraints
  -> repeat
```

Use scalar bracketed methods inside a stage when a suitable monotonic knob exists. Use multi-objective search for coupled controls.

### Stage G: Final validation

Freeze the candidate and evaluate it with independent seeds and a substantially larger sample. Do not select and certify on the same random streams.

---

## 16. Logging and Reproducibility

Every evaluation record should include:

```text
timestamp
candidate ID
parent candidate ID
search phase
complete normalized configuration
actual reel strips
generator seed
simulation seed
spin count
all target metrics
component RTPs
spin-return standard deviation
standard errors/confidence intervals
constraint violations
objective score
code/version identifier
```

JSONL is convenient for append-only search logs. Store full final candidates separately in human-readable configuration files.

Never log only aggregate grammar settings when the generated reel strips are random. The actual strips are part of the result.

---

## 17. Acceptance Criteria

A final candidate should be accepted only if:

- Every hard target is within its specified tolerance.
- Statistical uncertainty is sufficiently smaller than the tolerance.
- Component RTPs sum to total RTP under the documented accounting rules.
- Required trigger and hit-rate definitions are satisfied.
- Reel and weight constraints pass deterministic validation.
- Maximum-win and exposure limits pass.
- Results reproduce from saved configuration and seeds.
- Independent validation agrees with the search result.
- No target is met only because of early rounding or simulation noise.
- The complete win distribution remains consistent with the intended game experience.

---

## 18. Summary of Method Selection

| Problem | Preferred method |
|---|---|
| One continuous monotonic parameter | Bisection, Brent, or secant |
| Bracket not yet known | Directional probing and geometric expansion |
| Noisy scalar simulation | Common random numbers plus uncertainty-aware search |
| Small integer-weight table | Enumeration or local integer search |
| Several continuous parameters | Coordinate descent, CMA-ES, or Bayesian optimization |
| Coupled targets | Weighted constrained or Pareto optimization |
| Reel symbol counts | Integer/combinatorial optimization |
| Reel ordering | Simulated annealing, evolutionary search, or local mutations |
| Plausible reel generation | Constrained Markov/grammar proposal generator |
| Manageable reel state space | Exact enumeration |
| Persistent dynamic feature | Markov/DP analysis or seeded simulation |

The overall strategy is not to find one universal RTP knob. It is to construct a hierarchy of controls, solve the least-coupled targets first, use the appropriate search method for each parameter type, and repeatedly validate the full game as those partial solutions are combined.
