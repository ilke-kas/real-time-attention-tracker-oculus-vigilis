# Oculus Vigilis:Real-Time Attention Model

“Oculus Vigilis” aims  to help people to overcome their self-perceived attention problems such as ADHD or hyperactivity by giving a real-time feedback of the attention levels of one user using their camera records, especially during online meetings and learning processes.

## Getting Started

These instructions will give you a copy of the project up and running on
your local machine for development and testing purposes. See deployment
for notes on deploying the project on a live system.

### Prerequisites

Requirements for the software and other tools to build, test and push 
- For model training [Anaconda](https://jupyter.org/) or [Google Colab](https://colab.research.google.com/)
- High Performance GPUs like A100, NVIDIA GE3600 

## Attention Model
### Installing
After clonning this repository follow these steps:
A step by step series of creating necessary environments to run the attention model.
1. Open Anaconda Prompt
2. Create a virtual environment

        conda create --name attention_env -c anaconda python=3.11.0

3. Activate the environment

        conda activate attention_env
4. Install Jupyter Notebook

        conda install jupyter

5. Install these modules

        conda install tensorflow-gpu
        conda install pytorch torchvision torchaudio cudatoolkit -c pytorch
        conda pandas matplotlib scikit-learn torchmetrics tqdm seaborn pytorch-lightning tensorboard
        
7. Open Jupyter Notebook

        jupyter notebook
   
9. Go to the directory that the github files you cloned through jupyter notebook UI.

10. Go to DatasetPreperation file and open Attention_Model.ipynb

11. When you run the first line of the code

            !nvidia-sim
    Be sure that


End with an example of getting some data out of the system or using it
for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Sample Tests

Explain what these tests test and why

    Give an example

### Style test

Checks if the best practices and the right coding style has been used.

    Give an example

## Deployment

Add additional notes to deploy this on a live system

## Built With

  - [Contributor Covenant](https://www.contributor-covenant.org/) - Used
    for the Code of Conduct
  - [Creative Commons](https://creativecommons.org/) - Used to choose
    the license

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code
of conduct, and the process for submitting pull requests to us.

## Versioning

We use [Semantic Versioning](http://semver.org/) for versioning. For the versions
available, see the [tags on this
repository](https://github.com/PurpleBooth/a-good-readme-template/tags).

## Authors

  - **Billie Thompson** - *Provided README Template* -
    [PurpleBooth](https://github.com/PurpleBooth)

See also the list of
[contributors](https://github.com/PurpleBooth/a-good-readme-template/contributors)
who participated in this project.

## License

This project is licensed under the [CC0 1.0 Universal](LICENSE.md)
Creative Commons License - see the [LICENSE.md](LICENSE.md) file for
details

## Acknowledgments

  - Hat tip to anyone whose code is used
  - Inspiration
  - etc
