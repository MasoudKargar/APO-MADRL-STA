## install finrl library if not installed

import pkg_resources
import pip
installedPackages = {pkg.key for pkg in pkg_resources.working_set} # list of installed packages

if 'finrl' not in installedPackages:
#     %pip install git+https://github.com/AI4Finance-LLC/FinRL-Library.git #clone the FinRL library to install required packages

"""----
### 1.2. Check if the additional Required packages are present. If not, install them

* Yahoo Finance API
* pandas
* numpy
* matplotlib
* stockstats
* OpenAI gym
* stable-baselines
* tensorflow
* pyfolio
* ta
* PyPortfolioOpt
"""

# Commented out IPython magic to ensure Python compatibility.
required = {'yfinance', 'pandas','numpy', 'matplotlib', 'stockstats','stable-baselines',
            'gym','tensorflow','pyfolio', 'ta', 'PyPortfolioOpt'}
missing = required - installedPackages
if missing:
#     %pip install yfinance
#     %pip install pandas
#     %pip install numpy
#     %pip install matplotlib
#     %pip install stockstats
#     %pip install gym
#     %pip install tensorflow
#     %pip install git+https://github.com/quantopian/pyfolio
#     %pip install ta
#     %pip install PyPortfolioOpt