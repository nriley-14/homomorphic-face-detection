"""Setup script for faceid package."""

from setuptools import setup, find_packages

setup(
    name="faceid",
    version="1.0.0",
    description="Privacy-First Face Presence with Homomorphic Encryption",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "insightface==0.7.3",
        "numpy==2.2.4",
        "onnxruntime-gpu==1.21.1",
        "opencv-python==4.11.0.86",
        "tenseal==0.3.15",
        "pyyaml==6.0.2",
    ],
    entry_points={
        "console_scripts": [
            "faceid-enroll=faceid.enrollment_hub:main",
            "faceid-camera=faceid.camera_node:main",
            "faceid-server=faceid.he_server:main",
        ],
    },
)
