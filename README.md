# **NeurFlow: Interpreting Neural Networks through Neuron Groups and Functional Interactions (ICLR 2025)**
A framework that examines groups of critical neurons and their functional interactions that significantly influence model behavior.

## **Table of Contents**
- [Installation](#installation)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## **Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/tue147/neurflow.git
cd neurflow
```

## **Usage**

### **1. Make the result directory**

```bash
mkdir -p results
```

### **2. Create conda environment**

```bash
conda env create --name neurflow --file=environments.yml
```

### **3. Run the framework**

```bash
cd scripts
bash run.sh
```
### **4. Visualize the result**

Open notebooks for applications.
```
neurflow/
│── notebooks/   
    │── Debug_model.ipynb
    │── Explaining_concept.ipynb
    │── Visualizing_circuit.ipynb
```


## **Troubleshooting**

### **1. Environment setup error**

If you can't download choldate using environment.yml or `pip install git+git://github.com/jcrudy/choldate.git`:
- Try:
- ```bash
  git config --global url."https://".insteadOf git://
  pip install git+git://github.com/jcrudy/choldate.git
  ```

## **License**

This repository is licensed under the [MIT License](LICENSE).
  


