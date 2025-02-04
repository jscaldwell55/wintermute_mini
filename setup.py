from setuptools import setup, find_packages

setup(
    name="wintermute-cli",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click",
        "httpx",
        "rich",
        "pydantic"
    ],
    entry_points={
        'console_scripts': [
            'wintermute=cli.cli:cli',
        ],
    },
)