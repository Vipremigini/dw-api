from setuptools import setup, find_packages

setup(
    name="deepwriting",
    version="1.0.0",
    description="DeepWriting: Deep Generative Handwriting Synthesis & Digital Ink Editing",
    author="Emre Aksan, Fabrizio Pece, Otmar Hilliges",
    url="https://github.com/emreaksan/deepwriting",
    packages=find_packages(),
    py_modules=[
        "compat",
        "config",
        "dataset_hw",
        "tf_dataset_hw",
        "tf_models_hw",
        "tf_evaluate_hw",
        "tf_train_hw",
        "utils_hw",
        "visualize_hw",
        "synthesize",
    ],
    install_requires=[
        "tensorflow>=2.15.0",
        "scipy>=1.2.1",
        "numpy<2.0.0",
        "matplotlib>=3.2.0",
        "svgwrite>=1.4.0",
        "scikit-learn>=0.23.0",
        "opencv-python>=4.8.0",
        "Pillow>=9.0.0",
        "imageio>=2.30.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
