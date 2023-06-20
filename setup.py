import os
from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(name='Ashaar',
      version='0.0.2',
      url='https://github.com/ARBML/Ashaar',
      discription="Arabic poetry analysis and generation library",
      long_description=readme,
      long_description_content_type='text/markdown',
      author_email='arabicmachinelearning@gmail.com',
      license='MIT',
      packages=['Ashaar'],
      install_requires=required,
      python_requires=">=3.9",
      include_package_data=True,
      zip_safe=False,
      )
