from setuptools import setup

setup(name='dawis',
      version='0.0.1',
      description='Detection Algorithm with Wavelet for Intracluster light Studies',
      url='https://github.com/aellien/dawis.git',
      author='Amaël Ellien',
      author_email='a.r.j.ellien@uva.nl',
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
      zip_safe=False)
