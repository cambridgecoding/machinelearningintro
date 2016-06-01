# Machine Learning Setup Instructions

## Install Python

Install Anaconda (**Python 2.7**) from:  [https://www.continuum.io/downloads](https://www.continuum.io/downloads)
This includes python 2.7.9 and the necessary libraries we will be using: "numpy", "scipy" and "scikit-learn"

## Install Packages with pip

Installing required packages using "pip"

Open your terminal and check whether you have the "pip" function installed by typing pip (and enter).
If you do not have pip installed, check the link: [https://pip.pypa.io/en/latest/installing/](https://pip.pypa.io/en/latest/installing/) (If installing via the terminal/command line, ensure you are in the directory where you have downloaded the file "get-pip" or if using chrome right-click on the link to download, save to desktop, and simply double click on the executable).

You may need to use `sudo pip install` (for OSX, *nix, etc) or run your command shell as Administrator (for Windows) to be able to perform the installation of the following individual packages:

    (sudo) pip install Plotly

If you already have any of the previously-mentioned libraries installed, you can update them to a newer version using the syntax:

    pip install <package> --upgrade

where `<package>` can be any of the libraries mentioned above.


## Install git

Install git if you don't have it: [http://git-scm.com/](http://git-scm.com/)


##  Sign up for a GitHub

Sign up for a GitHub account or sign in if you have one: [github.com](https://github.com)


## Clone or download the code from the CCA GitHub repository

You can create a copy of the provided code on your local machine by using the "git clone" command on your console: 

    git clone https://github.com/cambridgecoding/machinelearningintro.git

Alternatively, click on the "Download ZIP" button under [https://github.com/cambridgecoding/machinelearningintro](https://github.com/cambridgecoding/machinelearningintro)


## Finalise the setup

Open and run the "[load_libraries.ipynb](https://github.com/cambridgecoding/machinelearningintro/blob/master/load_libraries.ipynb)" file, and check whether the libraries have been successfully loaded.

To execute the notebook, in your terminal run:

    ipython notebook load_libraries.ipynb

* You can run the notebook document step-by-step (one cell a time) by pressing **shift + enter**.
* You can run the whole notebook in a single step by clicking on the menu Cell -> Run All.
* To restart the kernel (i.e. the computational engine), click on the menu Kernel -> Restart. This can be useful to start over a computation from scratch (e.g. variables are deleted, open files are closed, etc...).
* Click on the menu Help -> User Interface Tour for an overview of the Jupyter Notebook App user interface.

