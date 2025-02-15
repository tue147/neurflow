# **Fidelity of interaction weight**

To run these experiments, first create directory:

```bash
cd exps/fidelity_of_weight/
mkdir -p correlation
```

## **1. Fidelity of neuron interaction weight**

Run the script.

```bash
bash correlation.sh
```

You can plot the results using the `Correlation_plot.ipynb`.

## **2. Comparison of attribution method**

Run the script.

```bash
bash correlation_all.sh
```

You can plot the results using the `Correlation_plot.ipynb`.

## **3. Neuron group relation aggregation**

Run the script.

```bash
bash correlation_compare_sum_and_mean.sh
```

You can plot the results using the `Correlation_plot.ipynb`.
