from setuptools import find_packages, setup

setup(
    name="EigenHunt",
    packages=[package for package in find_packages() if (package.startswith("discovery") or package.startswith("identification"))],
    install_requires=["gym", "numpy"],
    extras_require={
        "tests": [
            # Type check
            "pytype",
            # Lint code
            "flake8>=3.8",
            # Sort imports
            "isort>=5.0",
            # Reformat
            "black",
        ]
    },
    description="Gym env for fitting the CPG.",
    author="Andres Gonzalez",
    url="",
    author_email="j.andregon15@gmail.com",
    keywords="reinforcement-learning-algorithms reinforcement-learning machine-learning ",
    license="MIT",
    long_description="",
    long_description_content_type="text/markdown",
    version="0.7.0",
)

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
