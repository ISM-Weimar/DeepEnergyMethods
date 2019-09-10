Collocation code based on https://github.com/maziarraissi/PINNs 

Requires Tensorflow and pyDOE

### Recommended steps for running the collocation code using tensorflow with CPU support:

1.	Install Anaconda Python 3 branch by downloading it from https://www.anaconda.com/download/
If you already have an older version of Anaconda installed, it might be a good idea to update it by typing `conda update --all` at the “Anaconda prompt” or uninstall and reinstall it. You can leave the default options which do not require administrator access.

2.	Open the Anaconda Prompt from the Start Menu on Windows or by typing `conda activate base` on Linux.

3.	Create a tensorflow environment named tf. This is required in newer versions of Anaconda which come with Python 3.7 or higher as the current version of tensorflow requires Python 3.6. The command to create the environment and install tensorflow and its dependencies is:
`conda create --name tf tensorflow`

4.	Activate the tensorflow environment with the command:
`conda activate tf`

5.	Install the spyder IDE and some additional packages:
```
conda install spyder
conda install matplotlib
conda install -c conda-forge pydoe
```
Note: Although spyder and matplotlib are installed already in the base environment, they should be installed and run from the "tf" environment (in Python 3.6) to ensure that they work correctly with programs which use tensorflow.

6.	Start spyder from the Anaconda prompt:
`spyder`

7.	Open the file you would like to run e.g. `(…)DeepEnergyMethods/tensorflow_collocation/Poisson/Poisson2D_Dirichlet.py` and run it in the current console with the Green arrow button in the toolbar or by pressing F5

For subsequent runs, you can of course skip the steps 1, 3, and 5. 
