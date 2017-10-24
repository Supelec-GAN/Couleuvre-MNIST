TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

HEADERS += \
    headers/CSVFile.h \
    headers/functions.hpp \
    headers/mnist_reader.h \
    headers/neuralnetwork.hpp \
    headers/neuronlayer.hpp \
    headers/teacher.hpp

SOURCES += \
    sources/functions.cpp \
    sources/main.cpp \
    sources/mnist_reader.cpp \
    sources/neuralnetwork.cpp \
    sources/neuronlayer.cpp \
    sources/teacher.cpp \
    headers/neuralnetwork.inl
