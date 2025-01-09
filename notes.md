### Notes

1. This branch is for running the task in a fixed time regime
2. Let's run two scenario's,
   1. Hard task with more overlap
   2. Easy task / Normal task
   3. 10 hidden neurons compared to 50
3. For the fixed time regime, we modify the environment
   1. Rnn looks at the stimulus for 12 steps before making a decision
      1. we still store the output probabilites
      2. no penalty if it does not reach a decision, that workflow doesn't make sense
      3. reward only if correct decision reached at the end