# Method overview

The public reference environment receives:

- a list of discrete action domains, one domain for each design variable;
- an optional feasibility function for state-dependent action masking; and
- a callable terminal evaluator.

At each step the environment exposes only the feasible actions for the active
variable. The selected action is assigned to that variable and the next
variable becomes active. No expensive evaluator is called during partial
design construction. After the final assignment, the evaluator receives a
copy of the complete design exactly once and its scalar output becomes the
terminal reward.
