from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bift",
    version="0.1.0",
    author="BIFT Contributors",
    description="Biological-inspired Invariant Feature Transform for Multimodal Image Matching",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yyxgiser/BIFT",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
)
