# Example include file

CXX      := g++
CXXFLAGS := -O3
DEFINES  := -DNDEBUG -DFORTRAN_STRLEN_END
CXXFLAGS := -O3 -march=native -g
LDFLAGS  := -flto -O3 -lm
LIBS     := -lopenblas
