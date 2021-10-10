from setuptools import setup, find_packages

setup(name='keras_rl',
      version='1.0',
      description='A Basic setup for Reinforcement Learning using Keras',
      author='Rohit Gopalan',
      author_email='rohitgopalan1990@gmail.com',
      license='DST',
      packages=find_packages(),
      install_requires=['numpy', 'torch', 'gym', 'scikit-learn', 'joblib'],
      zip_safe=False)
