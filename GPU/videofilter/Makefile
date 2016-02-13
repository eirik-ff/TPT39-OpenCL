EXE=videofilter
CVINCFLAGS=`pkg-config --libs --cflags opencv`
CVLIBFLAGS=`pkg-config --libs opencv`
DBGFLAGS= -pg
FLAGS=-std=c++11 -O3
GCC=g++
SRCS=./videofilter.cpp
all:${EXE}
${EXE}: ${SRCS}
	${GCC} ${DBGFLAGS}  ${CVINCFLAGS} ${SRCS} ${CVLIBFLAGS} ${FLAGS} -o ${EXE}


profile:
	./${EXE}
	gprof ./${EXE}  gmon.out > prof.txt

run:${EXE}
	./${EXE}

clean:
	rm -rf ${EXE} *.o
