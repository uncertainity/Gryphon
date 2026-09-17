You are tuning the RTP/math configuration for this game.

A helper document containing general RTP-tuning techniques and recommended search strategies is available at:

`<PATH_TO_TUNING_MD>`

Use it as guidance where appropriate. You do not need to apply every technique in the document; choose methods that fit the current game and tuning variables.

## Scope

All code, scripts, intermediate configs, tuning logs, and tuning-specific utilities that you create must remain inside:

`Numba_Engine/Tuning/`

Do not modify unrelated parts of the repository unless it is necessary for the tuning process. If a change outside this directory is required, keep it minimal and clearly document why it was needed.

## Targets

The required tuning targets for this game are:

<INSERT TARGET RTP / HIT RATE / FEATURE FREQUENCY / OTHER CONSTRAINTS HERE>

Treat these values as the source of truth for the current tuning run.

## Workflow

Inspect the existing game implementation and determine which parameters actually control the requested metrics.

Use the techniques from the tuning guide where useful—for example exact evaluation, seeded simulation, scalar search, weight tuning, staged optimization, or multi-objective search.

Prefer reproducible experiments. Preserve useful candidate configurations, seeds, metrics, and tuning results so that the work can be resumed later rather than restarted from scratch.

Do not optimize blindly around an obviously incorrect metric or implementation. If you discover a likely bug, inconsistent counter, unreachable target, or important tuning constraint, record it clearly.

## Persistent Summary

Maintain a summary file inside `Numba_Engine/Tuning/` containing the latest state of the tuning work.

The summary should be updated whenever meaningful progress is made and should include:

* Current targets and tolerances.
* Current best configuration and measured metrics.
* Parameters that were changed and why.
* Techniques/search methods already attempted.
* Important observations about parameter sensitivity or coupling.
* Failed approaches or ranges that should not be repeated.
* Seeds / simulation sizes / evaluation settings needed to reproduce important results.
* Remaining gaps between the current configuration and the targets.
* Recommended next steps.

Assume that another tuning session may continue from this summary later. It should contain enough information to understand the current state without reconstructing the entire tuning history from the conversation.
