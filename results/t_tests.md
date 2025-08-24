**Paired t-tests: vs Baseline**

| Experiment                  | Mean Δ Test Acc (%) | t-stat | p-value |
| --------------------------- | ------------------: | -----: | ------: |
| cifar10\_abs\_dropout\_2em2 |          **+0.371** |  4.747 |  0.0003 |
| cifar10\_abs\_dropout\_1em2 |              +0.166 |  1.976 |  0.0681 |
| cifar10\_abs\_dropout\_5em3 |              +0.044 |  0.418 |   0.683 |

---

**Paired t-tests: vs Std Dropout**

| Experiment                  | Mean Δ Test Acc (%) | t-stat | p-value |
| --------------------------- | ------------------: | -----: | ------: |
| cifar10\_abs\_dropout\_2em2 |          **+0.371** |  4.747 |  0.0003 |
| cifar10\_abs\_dropout\_1em2 |              +0.166 |  1.976 |  0.0681 |
| cifar10\_abs\_dropout\_5em3 |              +0.044 |  0.418 |   0.683 |

---

**Train Loss**

| Experiment                  | Mean    | Variance    |
| --------------------------- | ------- | ----------- |
| cifar10\_baseline           | 0.00858 | 0.000004201 |
| cifar10\_std\_dropout\_1em2 | 0.00753 | 0.000002375 |
| cifar10\_std\_dropout\_2em2 | 0.00636 | 0.000003269 |
| cifar10\_std\_dropout\_5em3 | 0.00826 | 0.000004645 |
| cifar10\_abs\_dropout\_1em2 | 0.00716 | 0.000003136 |
| cifar10\_abs\_dropout\_2em2 | 0.00771 | 0.000003396 |
| cifar10\_abs\_dropout\_5em3 | 0.00947 | 0.000009028 |

---

**Test Loss**

| Experiment                  | Mean    | Variance    |
| --------------------------- | ------- | ----------- |
| cifar10\_baseline           | 0.45308 | 0.000141994 |
| cifar10\_std\_dropout\_1em2 | 0.45434 | 0.000824356 |
| cifar10\_std\_dropout\_2em2 | 0.43334 | 0.000587397 |
| cifar10\_std\_dropout\_5em3 | 0.45024 | 0.000451026 |
| cifar10\_abs\_dropout\_1em2 | 0.45585 | 0.000419257 |
| cifar10\_abs\_dropout\_2em2 | 0.44825 | 0.000248603 |
| cifar10\_abs\_dropout\_5em3 | 0.45597 | 0.000702034 |

---

**Train Accuracy (%)**

| Experiment                  | Mean   | Variance |
| --------------------------- | ------ | -------- |
| cifar10\_baseline           | 99.718 | 0.004770 |
| cifar10\_std\_dropout\_1em2 | 99.751 | 0.003309 |
| cifar10\_std\_dropout\_2em2 | 99.790 | 0.003859 |
| cifar10\_std\_dropout\_5em3 | 99.724 | 0.003976 |
| cifar10\_abs\_dropout\_1em2 | 99.763 | 0.003772 |
| cifar10\_abs\_dropout\_2em2 | 99.740 | 0.003968 |
| cifar10\_abs\_dropout\_5em3 | 99.682 | 0.009717 |

---

**Test Accuracy (%)**

| Experiment                  | Mean   | Variance |
| --------------------------- | ------ | -------- |
| cifar10\_baseline           | 92.823 | 0.023452 |
| cifar10\_std\_dropout\_1em2 | 92.687 | 0.087392 |
| cifar10\_std\_dropout\_2em2 | 92.910 | 0.057514 |
| cifar10\_std\_dropout\_5em3 | 92.706 | 0.051126 |
| cifar10\_abs\_dropout\_1em2 | 92.849 | 0.065555 |
| cifar10\_abs\_dropout\_2em2 | 93.051 | 0.039084 |
| cifar10\_abs\_dropout\_5em3 | 92.726 | 0.070983 |
