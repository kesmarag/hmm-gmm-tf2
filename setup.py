from setuptools import setup

setup(name='kesmarag-hmm-gmm-tf2',
      version='0.3.0',
      description='HMM class with GMM emission distributions',
      author='Costas Smaragdakis',
      author_email='kesmarag@gmail.com',
      url='https://github.com/kesmarag/hmm-gmm-tf2',
      packages=['kesmarag.hmm'],
      package_dir={'kesmarag.hmm': './'},
      install_requires=['tensorflow',
                        'tensorflow-probability',
                        'scikit-learn>=0.18.1',
                        'numpy>=1.12.1'], )
