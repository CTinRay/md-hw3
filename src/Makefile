CXXFLAGS = -std=c++11 -Wall -O3 -mtune=native -Ofast -flto -funroll-loops -lboost_program_options
# CXXFLAGS = -std=c++11 -Wall
DBGFLAGS = -g3

OBJS = codebook.o nmtf.o utils.o 

.PHONY: clean

%.o: ../%.cpp
	$(CXX) -c $(CXXFLAGS) $(DBGFLAGS) $<

codebook: $(OBJS)
	$(CXX) $(CXXFLAGS) $(DBGFLAGS) $(OBJS) -o $@

clean:
	rm $(OBJS)
