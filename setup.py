"""MossSpider provides an implementation of the targeted maximum likelihood estimator for network-dependent data
(network-TMLE).

To install the `mossspider`, use the following command

$ python -m pip install mossspider

"""

from setuptools import setup

exec(compile(open('mossspider/version.py').read(),
             'mossspider/version.py', 'exec'))

with open("README.md") as f:
    descript = f.read()


setup(name='mossspider',
      version=__version__,
      description='Targeted maximum likelihood estimation for network data',
      keywords='tmle network',
      packages=['mossspider',
                ],
      include_package_data=True,
      license='MIT',
      author='Paul Zivich',
      author_email='zivich.5@gmail.com',
      url='https://github.com/pzivich/',
      classifiers=['Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8',
                   'Programming Language :: Python :: 3.9',
                   'Programming Language :: Python :: 3.10',
                   ],
      install_requires=['numpy',
                        'scipy',
                        'pandas',
                        'networkx>=2.0.0',
                        'matplotlib',
                        'statsmodels',
                        'patsy'
                        ],
      long_description=descript,
      long_description_content_type="text/markdown",
      )
