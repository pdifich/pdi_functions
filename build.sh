#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "Usage $0 file.cpp" >&2
	echo "Construye el ejecutable file.bin" >&2
	exit 1
fi

file=$1
g++ "${file}" -W{all,extra,pedantic} -c -ggdb -std=c++11 -o "${file%.cpp}.o"
g++ "${file%.cpp}.o" $(pkg-config --libs opencv) -o "${file%.cpp}.bin"
