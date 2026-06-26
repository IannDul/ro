from setuptools import find_packages, setup


package_name = "vision_module_camera_selector"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="Ian Dul",
    maintainer_email="dulyoung4d@gmail.com",
    description="Best-view camera selection node for vision-module.",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "camera_selector_node = vision_module_camera_selector.camera_selector_node:main",
        ],
    },
)
