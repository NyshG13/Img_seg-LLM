from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'final_pipeline'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
	(os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='naysha',
    maintainer_email='nayshagupta27@gmail.com',
    description='Node for ML and ROS2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
		'camera = final_pipeline.camera_integ:main',
		'img_segmentation = final_pipeline.grounded_sam2_tracking_camera_with_continuous_id:main',
		'pipeline_node = final_pipeline.new_pipeline:main',
		'position = final_pipeline.position_data:main',
        ],
    },
)
