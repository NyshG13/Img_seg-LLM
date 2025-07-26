from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'final_pipeline'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(include=[package_name, package_name + '.*']),
    data_files=[
        # Resource index for ament
        ('share/ament_index/resource_index/packages',
            ['resource/final_pipeline']),

        # Package metadata
        ('share/' + package_name, ['package.xml']),

        # Install service definitions
        # (os.path.join('share', package_name, 'services'), glob('services/*.srv')),

        # Install launch files if added later
        # (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'configs'), glob('final_pipeline/configs/**/*.yaml', recursive=True)),
        (os.path.join('share', package_name, 'checkpoints'), glob('final_pipeline/checkpoints/**/*.pt', recursive=True)),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='naysha',
    maintainer_email='nayshagupta27@gmail.com',
    description='Final pipeline with ML and ROS2 services',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'new_pipeline = final_pipeline.new_pipeline:main',
            # 'camera_integ = final_pipeline.camera_integ:main',
            # 'gps_tracking = final_pipeline.gps_tracking:main',
        ],
    },
)
