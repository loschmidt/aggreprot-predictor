import setuptools
import nnap

setuptools.setup(
    name="aggreprot-predictor",
    version=nnap.__version__,
    maintainer="Simeon Borko",
    maintainer_email="simeon.borko@recetox.muni.cz",
    description="NNAP: Neural Network based Amyloid Predictor",
    long_description=None,
    long_description_content_type="text/markdown",
    url="https://git.loschmidt.cz/aggreprot/aggreprot-predictor",
    packages=["nnap"],
    package_data={"nnap": [
        nnap.SEQ_FILE,
        nnap.SEQ_MODEL_DIR + "/*",
    ]},
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: Console",
    ],
    python_requires=">=3.8,<3.9",
    install_requires=[
        "tqdm",
        "click==8.1.3",
        "scikit-learn",
        "pandas==1.4.2",
        "tensorflow==2.8",
        "tabulate==0.8.9",
        "protobuf==3.20.3",  # higher versions cause errors, the message says to install 3.20.x
        "biopython",
    ],
    entry_points={
        'console_scripts': [
            'nnap=nnap.cli:main',
            'aggreprot-predictor=nnap.cli:main',
        ],
    }
)
