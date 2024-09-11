
# Investigating Cost-effective Acoustic Detection of Drone Presence

This repository contains the code and dataset for the MSc project "Investigating Cost-effective Acoustic Detection of Drone Presence" by Yaoxiang Yu at the University of Southampton.

## Project Overview

The goal of this project is to explore cost-effective solutions for detecting the presence of drones using acoustic data. By applying machine learning techniques to acoustic signals, this project aims to develop a real-time acoustic drone detection system deployed on a Raspberry Pi 5 (8GB).

## Requirements

The code is written in Python and relies on the following libraries:

- `torch==2.5.0`  
- `numpy==1.23.5`  
- `matplotlib==3.7.2`

Ensure you have these libraries installed before running the code. You can install the required dependencies using the following command:

```bash
pip install torch==2.5.0 numpy==1.23.5 matplotlib==3.7.2
```

## Dataset

The dataset used in this project contains acoustic recordings for drone detection. It is included in this repository or can be accessed via the provided link (if applicable). The drone sound is downloaded from https://github.com/saraalemadi/DroneAudioDataset. The Scene noise and other types of noise can be downloaded from DCASE.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/asheryuyuyu/MSc_Project.git
cd MSc_Project
```

2. Install the required dependencies as mentioned above.

3. Run the code for training.

## Raspberry Pi Deployment

For deployment on a Raspberry Pi 5 (8GB), the following additional libraries are required:

- `torch==2.2.1`  
- `sounddevice==0.4.7`  
- `gpiozero==2.0.1`

You can install these libraries using the following command:

```bash
pip install torch==2.2.1 sounddevice==0.4.7 gpiozero==2.0.1
```

On the Raspberry Pi, the detection program can be run with:

```bash
python3 rasp2.py
```
## Contact

For any questions or issues, feel free to contact:

- **Yaoxiang Yu** yy4u23@soton.ac.uk
