RMDIR /S /Q build
RMDIR /S /Q dist
RMDIR /S /Q easyNeuron.egg-info

py setup.py sdist bdist_wheel
twine upload dist/*