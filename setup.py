from setuptools import find_packages, setup

package_name = 'diffdef_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='britton',
    maintainer_email='brittonty@gmail.com',
    description='ROS 2 node that predicts a goal point cloud from start and context '
                'point clouds with DiffDef',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'diffdef_node = diffdef_pkg.diffdef_node:main',
        ],
    },
)
