from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

setup(
    name="gvpy",
    version="0.2.0",
    author="Gunnar Voet",
    author_email="gvoet@ucsd.edu",
    url="https://github.com/gunnarvoet/gvpy",
    license="GNU GPL v3",
    # Description
    description="Library of python modules for data analysis and visualization",
    long_description=f"{readme}\n\n{history}",
    long_description_content_type="text/x-rst",
    # Requirements
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "gsw",
        "scipy",
        "xarray",
        "matplotlib",
        "seabird",
        "munch",
        "pandas",
        "IPython",
        "requests",
        "mixsea",
        "lat-lon-parser",
        "cartopy",
    ],
    extras_require={
        "cartopy": ["cartopy"],  # install these with: pip install gvpy[cartopy]
    },
    # Packaging
    packages=find_packages(include=["gvpy", "gvpy.*"], exclude=["*.tests"]),
    zip_safe=False,
    platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
    # Metadata
    project_urls={"Documentation": "https://github.com/gunnarvoet/gvpy"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
