import setuptools


setuptools.setup(name='pyzgy',
                 author='equinor',
                 description='Convenience wrapper around OpenZGY Python implementation',
                 long_description='Convenience wrapper around OpenZGY Python implementation which enables reading ZGY files with a syntax familiar to users of segyio.',
                 url='https://github.com/equinor/pyzgy',

                 use_scm_version=True,
                 install_requires=['numpy', 'segyio'],
                 setup_requires=['setuptools', 'setuptools_scm'],
                 packages=['pyzgy', 'openzgy', 'openzgy.impl']
                 )
