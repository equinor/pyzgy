import setuptools

setuptools.setup(name='zgyio',
                 author='equinor',
                 description='Convenience wrapper around OpenZGY Python implementation',
                 long_description='Convenience wrapper around OpenZGY Python implementation which enables reading ZGY files with a syntax familiar to users of segyio.',
                 url='https://github.com/equinor/zgyio',

                 version='0.0.0',
                 install_requires=['numpy'],
                 setup_requires=['setuptools'],
                 packages=['zgyio', 'openzgy']
                 )
