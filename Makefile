TOPSRCDIR := .
include $(TOPSRCDIR)/make.inc

CXXFLAGS += -std=c++17

.PHONY: all bench test clean

OBJS := generator.o trsv.o

# To get coverage reports, cd analysis && ./coverage.sh
ifeq ($(coverage), 1)
	CXXFLAGS += --coverage
	LDFLAGS += --coverage
endif

all: bench test

test: test.o $(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) test.o -o test $(LIBS)

bench: bench.o $(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) bench.o -o bench $(LIBS)

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $(DEFINES) -c $< -o $@

clean: 
	rm -f test bench *.o *.gcda *.gcno *.gcov
