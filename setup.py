import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="python-RRT", # Replace with your own username
    version="0.0.1",
    author="Adwait Naik",
    author_email="adwaitnaik2@gmail.com",
    description="A python package for RRT motion planning algorithm",
    long_description="sampling based motion planning algorithms like RRT(Rapidly exploring Random Tree) and PRM(Probabilistic Roadmap) in 2D configuration space with pygame.",
    long_description_content_type="text/markdown",
    url="https://github.com/addy1997/python-RRT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
