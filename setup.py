import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bioanalyse",
    version="0.0.1.dev1",

    author="SC van Nostrand",
    author_email="scvannost@gmail.com",
    description="Handles scRNAseq processing from a count matrix to clustering and visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/pypa/sampleproject",
    license='GPLv3',
    classifiers=[
        "Development Status :: 3 - Alpha"
        "Intended Audience :: Science/Research"
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ],
    keywords='scrna biology compbio rnaseq scrnaseq processing',
    python_requires='~=3.6',

    packages=['bioanalyse'],
    package_data={'bioanalyse':['HK_homo.csv', 'HK_mouse.csv']},
    install_requires=['numpy', 'scipy', 'matplotlib', 'pandas', 'sklearn', 'mpl_toolkits', 'umap', 'frozendict', 'igraph'],
)