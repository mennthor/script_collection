from setuptools import setup, find_packages

setup(
    name="script_collection",
    version="0.1",
    description="Collection of python scripts installed as cmd entrypoints",
    url="https://github.com/mennthor/script_collection",
    author="Thorben Menne",
    author_email="thorben.menne@tu-dortmund.de",
    license="MIT",
    install_requires=[],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "replace_spaces=script_collection.replace_spaces:main"
        ],
    }
)
