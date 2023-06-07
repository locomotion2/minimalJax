#!/bin/bash
CONAN_CACHE="/volume/conan_cache/$USER/.conan/data"

LN_MANAGER_PATH="$CONAN_CACHE/links_and_nodes_manager/2.1.7/common/stable/package/b826c19cddf0a5cde0ca67265e78b65ffb3265ab/lib"
LN_PYTHON_PATH="$CONAN_CACHE/links_and_nodes_python/2.1.2/common/stable/package/b26b3132f92112240d5c7fae72ee72000c94ada7/lib"
LN_LIB_PATH="$CONAN_CACHE/liblinks_and_nodes/2.1.6/common/stable/package/c3fc377a719563518ee16c403eeedfaf166c0e92/lib"
LN_BASE_PATH="$CONAN_CACHE/links_and_nodes_base_python/2.1.2/common/stable/package/f6c004d398a9c99a9c080c8f3502ebea44092434/lib"
BOOST_PYTHON_PATH="$CONAN_CACHE/boost_python/1.66.0/3rdparty/stable/package/b247a45c04d9eb070db983c343452dddb24c1671/lib"
BOOST_PATH="$CONAN_CACHE/boost/1.66.0/3rdparty/stable/package/4a16ac2011ceae2c50d93d495b3a1643f027614a/lib"

# Hack to have LN working with python3.7
cp -R $LN_PYTHON_PATH/python/site-packages/links_and_nodes/linux-x86_64-3.6/ $LN_PYTHON_PATH/python/site-packages/links_and_nodes/linux-x86_64-3.9

export LD_LIBRARY_PATH="$LN_PYTHON_PATH:$LN_LIB_PATH:$LN_BASE_PATH:$BOOST_PYTHON_PATH:$BOOST_PATH:$LD_LIBRARY_PATH"
export PYTHONPATH="$LN_PYTHON_PATH/python/site-packages:$LN_BASE_PATH/python/site-packages:$LN_MANAGER_PATH/python/site-packages:$PYTHONPATH"
