#/bin/bash

rm Obser*
sphinx-apidoc -f -o . ../

make clean
make html

