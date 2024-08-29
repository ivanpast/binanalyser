from setuptools import setup, find_packages

setup(
    name="interactive_binning",
    version="0.1",
    author="Ivan Pastor",
    author_email="ivanpastorsanz@gmail.com",
    description="A package for interactive binning and statistical analysis in Streamlit.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/ivanpast/binanalyser",  # Replace with your GitHub repo
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "optbinning",
        "scipy",
        "matplotlib",
        "seaborn",
        "xlsxwriter",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
