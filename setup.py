import setuptools

setuptools.setup(
    name="exploratory_data_analysis",
    version="0.1.0",
    description="Exploratory Data Analysis Tool Suite",
    license="TODO",
    author="Constantin Gahr",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=["numpy", "matplotlib", "seaborn", "scipy"],
    setup_requires=["pytest-runner"],
    test_suite="pytest",
    tests_require=["pytest"],
)
