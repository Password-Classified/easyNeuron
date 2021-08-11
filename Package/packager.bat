RMDIR /S /Q build
RMDIR /S /Q dist
RMDIR /S /Q easyNeuron.egg-info

ECHO MAKE SURE YOU HAVE CREATED NEW VERSION NAME/NUMBER AND UPDATED __init__.py.

py setup.py sdist bdist_wheel
twine upload dist/*