from setuptools import setup

setup(name='dawis',
      version='0.0.7',
      description='Detection Algorithm with Wavelet for Intracluster light Studies',
      url='https://github.com/aellien/dawis.git',
      author='Amaël Ellien',
      author_email='amael.ellien@oca.eu',
      license='GNU General Public License v3.0',
      packages=['dawis'],
      install_requires=[
          'numpy',
          'matplotlib',
          'scipy',
          'astropy',
          'scikit-image',
          'ray'
      ],
      include_package_data=True,
      package_data={'': ['gallery/*.fits']},
      zip_safe=False)
