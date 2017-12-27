CC=gcc-5
all:
	${CC} -std=gnu99 -fopenmp -o mrf main.c plm.h plm.c clm.h clm.c plm_glasso.h plm_glasso.c sequence.h sequence.c util.h util.c model.h model.c -llbfgs -lm -O3
clean:
	rm -rf *.o
