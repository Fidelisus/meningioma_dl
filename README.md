# How to setup the repo on the cluster

# TODO update to the new deployment strategy

1. Prepare modules:

```
module add Python/3.9.5-GCCcore-8.2.0
module add PyTorch/1.9.0-foss-2019a
```

2. Create a Python 3.9 venv

3. Install requirements.txt . 
The versions of the packages have to be the same as there, it's a fragile compatibility with flwr.

4. Numpy hack to avoid a strange segmentation fault:

The error was:
```
Stack (most recent call first):
  File "<__array_function__ internals>", line 5 in count_nonzero
  File "/home/cir/lsobocinski/meningioma_dl/venv39/lib/python3.9/site-packages/numpy/lib/function_base.py", line 3954 in _quantile_is_valid
  File "/home/cir/lsobocinski/meningioma_dl/venv39/lib/python3.9/site-packages/numpy/lib/function_base.py", line 3928 in quantile
  File "<__array_function__ internals>", line 5 in quantile
  File "/home/cir/lsobocinski/meningioma_dl/meningioma_dl/visualizations/results_visualizations.py", line 311 in get_metric_linear_traces
```

Thus, you have to go to 
```
  File "/home/cir/lsobocinski/meningioma_dl/venv39/lib/python3.9/site-packages/numpy/lib/function_base.py", line 3954 in _quantile_is_valid
```

And change it to:

```
def _quantile_is_valid(q):
    # avoid expensive reductions, relevant for arrays with < O(1000) elements
    if True:
        for i in range(q.size):
            if q[i] < 0.0 or q[i] > 1.0:
                return False
    else:
        # faster than any()
        if np.count_nonzero(q < 0.0) or np.count_nonzero(q > 1.0):
            return False
    return True
```

You have to do it to avoid going to the problematic part of the code.

Then, you have to comment out the code there:

```
	Stack (most recent call first):
  File "<__array_function__ internals>", line 5 in any
  File "/home/cir/lsobocinski/meningioma_dl/venv39/lib/python3.9/site-packages/numpy/lib/function_base.py", line 408 in average
  File "<__array_function__ internals>", line 5 in average
  File "/home/cir/lsobocinski/meningioma_dl/meningioma_dl/run_federated_training.py", line 99 in _visualize_federated_learning_metrics
```
Comment out this code;

```
        if np.any(scl == 0.0):
            raise ZeroDivisionError(
                "Weights sum to zero, can't be normalized")
```

## PLEASE NOTE

Unfortunately, only nodes called `on*` work for FL, and additionally, only 2-3 jobs can be submitted per node... To set the node on which a job should be spawned, one can to either

```
--exclude cn5,cn6
```

or 

```
--nodelist on1
```

I observed that on3 node works the slowest for me. 