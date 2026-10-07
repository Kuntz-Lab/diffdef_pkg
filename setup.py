from setuptools import find_packages, setup

# Give console_scripts launchers a `#!/usr/bin/env python3` shebang, so they run
# on whichever Python is active (e.g. a venv with torch) rather than the one
# that ran colcon. setup.cfg's [build_scripts] covers a normal install, but
# `colcon build --symlink-install` uses `setup.py develop`, which ignores it.
try:
    from setuptools.command.easy_install import CommandSpec
    CommandSpec.from_environment = classmethod(
        lambda cls: cls.from_string('/usr/bin/env python3'))
except ImportError:
    pass

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
    description='ROS 2 nodes that predict a goal point cloud from start and context '
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
            'diffdef_service_node = diffdef_pkg.diffdef_service_node:main',
        ],
    },
)
