from setuptools import find_packages, setup

setup(
    name="custom_envs",
    packages=[package for package in find_packages() if package.startswith("custom_envs")],
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
    description="Custom Gym Environments.",
    author="Antonin Raffin",
    url="",
    author_email="antonin.raffin@dlr.de",
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
