from setuptools import setup

setup(name='kesmarag-hmm-gmm-tf2',
      version='0.1.0',
      description='HMM class with GMM emission distributions',
      author='Costas Smaragdakis',
      author_email='kesmarag@gmail.com',
      url='https://github.com/kesmarag/hmm-gmm-tf2',
      packages=['kesmarag.hmm'],
      package_dir={'kesmarag.hmm': './'},
      install_requires=['tensorflow>=2.0.0b1',
                        'tensorflow-probability==0.7.0',
                        'numpy>=1.12.1'], )
