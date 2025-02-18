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

### **1. Download ImageNet**
Download the 2012-version of ImageNet dataset (validation images), development-kit (development-kit task 1&2) on: **[Download Dataset]( https://image-net.org/)** 

Then unzip the validation set place them into a folder.
```
imagenet/
│── ILSVRC2012_devkit_t12.tar.gz
│── val
    │── data...
```

### **2. Make the result directory**

```bash
mkdir -p results
```

### **3. Create conda environment**

```bash
conda env create --name neurflow --file=environments.yml
```

### **4. Run the framework**

Copy your folder path to the dataset and place them in `DATA_DIR` in `scripts/run.sh`. Then run this in the terminal:

```bash
cd scripts
bash run.sh
```
### **5. Visualize the result**

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
  


