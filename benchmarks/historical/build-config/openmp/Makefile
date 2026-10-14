CC=gcc
CFLAGS=-O3 -fopenmp
LIBFLAGS=-lm -fopenmp

OBJDIR = obj

_OBJ = args.o data.o setup.o vtk.o boundary.o airfoil_omp.o
OBJ = $(patsubst %,$(OBJDIR)/%,$(_OBJ))

.PHONY: directories

all: directories airfoil_omp

obj/%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS) 

airfoil_omp: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBFLAGS) 

clean:
	rm -Rf $(OBJDIR)
	rm -f airfoil_omp

directories: $(OBJDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

