# How to setup the repo on the cluster

# TODO update to the new deployment strategy

1. Create a Python 3.13 venv

2. 

3. Install requirements.txt . 
The versions of the packages have to be the same as there, it's a fragile compatibility with flwr.

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