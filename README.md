# Search Transfer for Accelerator Design

The aim of this project was to explore whether knowledge from searching the design space of one neural network workload can be transferred to another workload. I used CarbonPATH to evaluate accelerator designs and Bayesian Optimization to search the design space.

The main question was whether a surrogate model trained while searching one model family could be reused to improve the search for another model family.

The experiment used two different model families. I first searched the accelerator design space independently for each model family and then compared this against transferring the surrogate learned from one model family to the other.

## First Results

### Transformer from scratch

Best after 50 initial evaluations: 30.3253

Final best: 10.8913

### CNN → Transformer transfer

Best after 50 initial evaluations: 18.9991

Final best: 15.7168

### Transfer improvement

Absolute: -4.8256

Percentage: -44.31%

The transferred surrogate did not outperform the independently trained search in the final result. In fact, the final best cost was higher by 4.83 (44.31%).

However, the transferred search started from a substantially better point: 18.9991 compared to 30.3253 after the initial 50 evaluations. This suggests that the surrogate learned from the CNN search contains useful information about the design space, but that this knowledge did not translate into a better final solution within the evaluation budget. 

This study seriously lacked a compute budget. What I would like to explore next is whether the transfer becomes more effective with different model pairs, larger evaluation budgets, or different methods for adapting the pretrained surrogate to the new workload.
