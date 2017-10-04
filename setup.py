from setuptools import find_packages, setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name="gvpy",
      version="0.1",
      description="Library of python modules for data analysis and visualization",
      author="Gunnar Voet",
      author_email='gvoet@ucsd.edu',
      platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
      license="GNU GPL v3",
      url="https://github.com/gunnarvoet/pythonlib",
      packages=find_packages(),
      )
