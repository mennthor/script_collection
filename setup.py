from setuptools import setup, find_packages

setup(
    name="script_collection",
    version="0.1",
    description="Collection of python scripts",
    url="https://github.com/mennthor/script_collection",
    author="Thorben Menne",
    author_email="thorben.menne@tu-dortmund.de",
    license="MIT",
    install_requires=[
        "numpy",
        "scipy",
        "healpy",
        "matplotlib",
        "tables",
        "scikit-learn",
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "replace_spaces=script_collection.cli.replace_spaces:_main",
            "logbook_new_week_template=script_collection.cli.logbook_new_week_template:_main",
        ],
    }
)
