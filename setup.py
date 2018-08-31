from setuptools import setup

with open("requirements.txt") as fh:
    requirements = [line.strip() for line in fh.readlines()]

setup(
    name="dd_widgets",
    version=0.1,
    description="IPython widgets for deepdetect",
    packages=["dd_widgets"],
    install_requires=requirements,
    python_requires=">=3.5",
)
