#!/bin/bash

# 当前脚本路径
SCRIPT_PATH=$(dirname "$(readlink -f "$0")")

if [ ! -d ${SCRIPT_PATH}/"external/systemc" ]; then
    cd  ${SCRIPT_PATH}/"external"
    rm -rf systemc-2.3.3/
    tar xzvf systemc-2.3.3.tar.gz
    cd systemc-2.3.3/
    mkdir build
    cd build/
    cmake ../ -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=${SCRIPT_PATH}/"external/systemc"
    make -j 8
    make install
    cd ../../
    rm -rf systemc-2.3.3
fi

rm -rf build

mkdir -p build
cd build
cmake ..
make -j8

./main

cd ..