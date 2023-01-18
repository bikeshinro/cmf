from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Metadata Python Package'
LONG_DESCRIPTION = 'Metadata framework storing AI metadata into MLMD'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="cmflib", 
        version=VERSION,
        author="Annmary Justine",
        author_email="annmary.roy@hpe.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=["ml-metadata==1.11.0",
                          "dvc[ssh]==2.27.0", "dvc[s3]==2.27.0", "pandas", "retrying", "pyarrow", "neo4j", "scikit-learn", "tabulate", "click", "minio", "paramiko"], # add any additional packages that
        # needs to be installed along with your package. Eg: 'caer'
        package_data={'':['*.sh']},
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: Linux :: Ubuntu",
        ],
        scripts=['cmflib/bin/cmf']
)
