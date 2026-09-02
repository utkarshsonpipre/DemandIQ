from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="demandiq",
    version="1.0.0",
    description="Automated demand forecasting & ML experimentation platform",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=Path("requirements.txt").read_text().splitlines(),
)
