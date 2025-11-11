CC=gcc-11
CPP=g++-11
CXX=g++-11
LD=g++-11
AR=ar

cxxflags = $(CPPFLAGS) $(CXXFLAGS) -I./cplusplus \
	-g -Wall -Wextra -Werror -std=c++20 -pthread -fPIC
ldflags = $(LDFLAGS)
#  -static
arflags = rc
LDLIBS = -L./cplusplus -lgrinder -pthread -lboost_program_options

PYFLAGS = -I/usr/include/python3.10 -L/usr/lib/python3.10 -lpython3.10

# g++ -fPIC -shared -I/usr/include/python3.10 -L/usr/lib/python3.10 mathlib.cpp -lpython3.10 -o libmath.so

sources = \

objects = $(sources:.cpp=.o)

$(info V is "$(V)")

ifeq ($(V),1)
	VCXX   = $(CXX) -c
	VCXXLD = $(CXX)
	VDEPS  = $(CXX) -MM
else
	VCXX   = @echo "  COMPILE $@" && $(CXX) -c
	VCXXLD = @echo "  LINK $@" && $(CXX)
	VDEPS  = @echo "  DEPENDS $@" && $(CXX) -MM
endif

all: agent-data.zip

ldaShell:
	docker build -t mylambda .

agent-data.zip:
	$(RM) -rf ./python
	pip3 install -r DataCollection/requirements.txt -t python/lib/python3.10/site-packages
	zip -r9 $@ python/

clean:
	$(RM) -rf ./python agent-data.zip

.PHONY: all clean
