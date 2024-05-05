# Oculus Vigilis:Real-Time Attention Model

“Oculus Vigilis” aims  to help people to overcome their self-perceived attention problems such as ADHD or hyperactivity by giving a real-time feedback of the attention levels of one user using their camera records, especially during online meetings and learning processes.

## Getting Started

These instructions will give you a copy of the project up and running on
your local machine for development and testing purposes. See deployment
for notes on deploying the project on a live system.

### Prerequisites

Requirements for the software and other tools to build, test and push 
- For model training [Google Colab](https://colab.research.google.com/)
- For model training High Performance GPUs like A100, NVIDIA GE3600
- For Application [Python 3.11.8](https://www.python.org/downloads/release/python-3118/)
- [Anaconda](https://www.anaconda.com/)

## Reproduce Attention Model 
(If you want you can skip this part since the model is already provided in this repository)
After clonning this repository follow these steps:
1. Go to this link [Google Colab Attention Model](https://colab.research.google.com/drive/1CC1o9xPHbpJg8zmE7Jf_lSAsPUoacrJj?usp=sharing)
2. Conect A100 GPU in order to run the code
3. Run the code
4. Copy the downloaded model to the clonned directory of this repository

## Application
1. Open Anaconda Prompt
2. Create Environment

       conda create --name attention_app
3. Activate Environment

       conda activate attention_app
4. Install Necessary modules through requirements.txt provided

       conda run py -m -r requirements.txt
5. Run the application

       conda run py UI.py

6. Click Start to start the session
7. Click Stop stop the recording
8. Close the program by hitting "x" on the right top of the window

## Authors

  - **Ilke Kas** - *PhD at ECSE* -
    [Ilke Kas Github](https://github.com/ilke-kas)

