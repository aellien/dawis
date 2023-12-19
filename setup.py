from setuptools import setup

setup(name='dawis',
      version='0.0.5',
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
      include_package_data=True,
      package_data={'': ['gallery/*.fits']},
      zip_safe=False)
