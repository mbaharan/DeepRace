# Deep Learning Reliability Awareness of Converters at the Edge (Deep RACE)
![POWERED BY TeCSAR](https://raw.githubusercontent.com/TeCSAR-UNCC/Deep_RACE/master/logo/tecsarPowerBy.png)

Deep RACE is a real-time reliability modeling and assessment of power semiconductor devices embedded into a wide range of smart power electronics systems. Deep RACE presents a scalable solution based on two deep learning algorithms that departures from classical and statistical modeling to deep learning-based data analytics. We leverage two states of the art algorithms for time series prediction named Long Short-Term Memory (LSTM) and Temporal Convolutional Networks (TCN) to aggregate reliability modeling of the power converters with the same underlying physics.

  
## Installation
You only need to clone the Deep RACE repository:
```bash
git clone https://github.com/mbaharan/DeepRace
```
The following instruction is for LSTM model. For TCN model you should first clone then switch the branch to TCN by:
```bash
cd DeepRace
git checkout TCN
```
Please follow the instruction in `README.md` of the `TCN` branch to train and run inference for the TCN model.

## Prerequisites
First make sure you have already installed pip3, Tkinter, and git tools:
``` bash
sudo apt install git python3-pip python3-tk
```
You should also install the following python packages:
```bash
sudo -H pip3 install tensorflow scipy matplotlib seaborn
```
## Training the network models
Change the path to the `DeepRACE` directory and run the `train.py`:
```bash
cd Deep_RACE
python3 ./train.py --test-dev 0
```
For seeing the list of devices, you can run the program with `--help`.


All the training models will be saved automatically in `./inference_models/` folder. You can load them by running the `inference.py` file.

### Prediction output
Running `inference.py` file generates and saves the prediction output in a text file. The file name is based on the selected MOSFET device number. For instance, a text file with the name of `./prediction_output/RUL_Dev_14.txt` will be generated for `dev#14`.

### Testing different MOSFET devices
You can test different devices from `dR11Devs.mat` by altering [this line](https://github.com/TeCSAR-UNCC/Deep_RACE/blob/68688f2b89a651f0985364c74c2ae949a696338b/train.py#L69) in the `./train.py`.

### RUL
For extracting the RUL at different time, you need to run the `inference.py` as follows:
```bash
./inference.py --test-dev 9 --rul-time 130 139 151 161 170 175 180 185 189
```


## Authors
* LSTM Model: Reza Baharani - [Personal webpage](https://rbaharani.com/)
* TCN  Model: Steven Furgurson

## License
Copyright (c) 2018, the University of North Carolina at Charlotte All rights reserved. - see the [LICENSE](https://raw.githubusercontent.com/TeCSAR-UNCC/Deep_RACE/master/LICENSE) file for details.
## Acknowledgments

The eleven Si-MOSFET ΔR<sub>ds(on)</sub> are extracted from NASA MOSFET Thermal Overstress Aging Data Set which is available [here](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/). Please cite their paper if you are going to use their data samples. Here is its BibTeX:
```
@article{celaya2011prognostics,
title={{Prognostics of power {MOSFET}s under thermal stress accelerated aging using data-driven and model-based methodologies}},
author={Celaya, Jose and Saxena, Abhinav and Saha, Sankalita and Goebel, Kai F},
year={2011}
}
```
