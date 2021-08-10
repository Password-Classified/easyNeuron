rmdir build
rmdir dist

cd src

rmdir easyNeuron.egg-info

cd ..

py setup.py sdist bdist_wheel
twine upload dist/*