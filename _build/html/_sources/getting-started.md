# Getting Started

In this course, we will be using Python and the Jupyter notebook environment to explore machine learning concepts. Below are the steps to get you set up and ready to go. There are two main options for setting up your environment: using Anaconda or using Google Colab.

We will focus on Anaconda here because it is the most common way to set up a Python environment for data science and machine learning. However, if you prefer to use Google Colab, you can simply [login to Google Colab](https://colab.research.google.com/) and create a new notebook. You can then upload or copy and paste the code from the Jupyter notebooks into your Colab notebook.

## Using Anaconda

1. Download the Anaconda installer for your operating system from the [Anaconda website](https://www.anaconda.com/products/distribution#download-section).
2. Follow the installation instructions for your operating system:
   - For Windows, run the installer and follow the prompts.
   - For macOS, open the downloaded `.pkg` file and follow the prompts.
   - For Linux, open a terminal and run the installer script.
3. After installation, open the Anaconda Navigator application.
4. In Anaconda Navigator, you can create a new environment by clicking on the "Environments" tab and then clicking "Create". Name your environment (e.g., `ml_course`) and select Python 3.x as the version.
5. Once the environment is created, you can install the necessary packages by clicking on the "Home" tab, selecting your environment, and then clicking "Install" next to the packages you need (e.g., `pandas`, `matplotlib`, `scikit-learn`).

You will need the following packages for this course:
- `pandas`: for data manipulation and analysis
- `matplotlib`: for plotting and visualization
- `scikit-learn`: for machine learning algorithms and tools
- `numpy`: for numerical operations
- `seaborn`: for statistical data visualization
- `ipykernel`: to run Jupyter notebooks in your environment
- `jupyter`: to run Jupyter notebooks
- `jupyter-lab`: for an enhanced Jupyter notebook interface

## Using a virtual environment

A [virtual environment](https://docs.python.org/3/tutorial/venv.html) is a self-contained directory that contains a Python installation for a particular version of Python, plus several additional packages. This allows you to manage dependencies for different projects separately. This can be useful when there are conflicting dependencies between different projects or when you want to ensure that your project runs with a specific version of a package.

### With Anaconda

If you would like to use a virtual environment with Anaconda, you can create one using the following steps:

1. Open Anaconda Prompt (or your terminal if you are on macOS or Linux).
2. Create a new environment with the following command:
   ```bash
   conda create --name ml_course
   ```
3. Activate the environment with:
   ```bash
    conda activate ml_course
    ```
4. Install the necessary packages with:
    ```bash
    conda install pandas matplotlib scikit-learn numpy seaborn ipykernel jupyter jupyter-lab
    ```
5. Add the kernel to Jupyter Notebook so you can use this environment:
    ```bash
    python -m ipykernel install --user --name=ml_course
    ```
6. You can now launch Jupyter Notebook from this environment by running:
    ```bash
    jupyter notebook
    ```
7. This will open a new tab in your web browser where you can create and run Jupyter notebooks.

### With venv

Alternatively, if you prefer to use Python's built-in `venv` module to create a virtual environment, you can follow these steps:

1. Open your terminal (Command Prompt on Windows, Terminal on macOS or Linux).
2. Navigate to the directory where you want to create your virtual environment.
3. Create a new virtual environment with the following command:
   ```bash
   python -m venv ml_course
   ```
4. Activate the virtual environment:
    - On Windows:
      ```bash
      ml_course\Scripts\activate
      ```
    - On macOS or Linux:
      ```bash
      source ml_course/bin/activate
      ```
5. Install the necessary packages with:
    ```bash
    pip install pandas matplotlib scikit-learn numpy seaborn ipykernel jupyter jupyter-lab
    ```
6. Add the kernel to Jupyter Notebook so you can use this environment:
    ```bash
    python -m ipykernel install --user --name=ml_course
    ```
7. You can now launch Jupyter Notebook or Lab by running:
    ```bash
    jupyter notebook
    ```
    or
    ```bash
    jupyter lab
    ```