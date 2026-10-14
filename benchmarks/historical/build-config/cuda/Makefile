CC=gcc
CXX=g++

CFLAGS=-O3
NVCC=nvcc

NVCCFLAGS=-O3 -arch=sm_75  -ccbin $(CXX)

LIBFLAGS=-lm

OBJDIR = obj

CUDA_SRCS = airfoil.cu data.cu
CUDA_OBJS = $(addprefix $(OBJDIR)/,$(CUDA_SRCS:.cu=.o))
CPU_SRCS = args.c setup.c vtk.c boundary.c
CPU_OBJS = $(addprefix $(OBJDIR)/,$(CPU_SRCS:.c=.o))
OBJ = $(CUDA_OBJS) $(CPU_OBJS)

.PHONY: directories

all: directories airfoil

obj/%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

$(OBJDIR)/airfoil.o: airfoil.cu
	$(NVCC) -c -o $@ $< $(NVCCFLAGS)

$(OBJDIR)/data.o: data.cu
	$(NVCC) -c -o $@ $< $(NVCCFLAGS)

airfoil: $(OBJ)
	$(NVCC) -o $@ $^ $(NVCCFLAGS) $(LIBFLAGS)

clean:
	rm -Rf $(OBJDIR)
	rm -f airfoil

directories: $(OBJDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)