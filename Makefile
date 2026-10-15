.DEFAULT_GOAL := cpu
.DELETE_ON_ERROR:

CC ?= cc
MPICC ?= mpicc
NVCC ?= nvcc
PYTHON ?= python3
BUILD_DIR ?= build
DEMO_DIR ?= out/demo
CFLAGS ?= -O3 -g -std=c11 -Wall -Wextra
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS := -lm
CUDA_ARCH ?= 75
NVCCFLAGS ?= -O3 -lineinfo -std=c++17 -arch=sm_$(CUDA_ARCH)

OPENMP_CFLAGS ?= -fopenmp
OPENMP_LDFLAGS ?= -fopenmp
ifeq ($(shell uname -s),Darwin)
ifneq ($(findstring clang,$(shell $(CC) --version 2>/dev/null)),)
LIBOMP_PREFIX ?= $(shell brew --prefix libomp 2>/dev/null)
OPENMP_CFLAGS = -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
OPENMP_LDFLAGS = -L$(LIBOMP_PREFIX)/lib -Wl,-rpath,$(LIBOMP_PREFIX)/lib -lomp
endif
endif

SUPPORT := args data setup vtk boundary
SERIAL_OBJ := $(addprefix $(BUILD_DIR)/serial/,$(addsuffix .o,$(SUPPORT)))
OMP_OBJ := $(addprefix $(BUILD_DIR)/openmp/,$(addsuffix .o,$(SUPPORT)))
MPI_OBJ := $(addprefix $(BUILD_DIR)/mpi/,$(addsuffix .o,$(SUPPORT)))
CUDA_OBJ := $(addprefix $(BUILD_DIR)/cuda/,$(addsuffix .o,args setup vtk boundary data airfoil))
SHARED_ARGS := $(BUILD_DIR)/serial/args.o $(BUILD_DIR)/openmp/args.o $(BUILD_DIR)/cuda/args.o
SHARED_VTK := $(BUILD_DIR)/serial/vtk.o $(BUILD_DIR)/openmp/vtk.o $(BUILD_DIR)/cuda/vtk.o
SHARED_DATA := $(BUILD_DIR)/serial/data.o $(BUILD_DIR)/openmp/data.o

.PHONY: all cpu serial openmp mpi cuda reference demo check check-historical validate info clean help
all: cpu
cpu: serial openmp
serial: $(BUILD_DIR)/airfoil_serial $(BUILD_DIR)/airfoil_serial_opt
openmp: $(BUILD_DIR)/airfoil_openmp $(BUILD_DIR)/airfoil_openmp_opt
mpi: $(BUILD_DIR)/airfoil_mpi
cuda: $(BUILD_DIR)/airfoil_cuda
reference: $(BUILD_DIR)/airfoil_reference

demo: serial
	$(PYTHON) scripts/demo.py --build-dir "$(BUILD_DIR)" --output-dir "$(DEMO_DIR)"

$(SHARED_ARGS): common/args.c $(wildcard serial/*.h openmp/*.h cuda/*.h)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(notdir $(@D)) -c $< -o $@

$(SHARED_VTK): common/vtk.c common/output_path.h $(wildcard serial/*.h openmp/*.h cuda/*.h)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(notdir $(@D)) -c $< -o $@

$(SHARED_DATA): common/data.c $(wildcard serial/*.h openmp/*.h)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(notdir $(@D)) -c $< -o $@

$(BUILD_DIR)/serial/%.o: serial/%.c $(wildcard serial/*.h) common/output_path.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -Iserial -c $< -o $@

$(BUILD_DIR)/openmp/%.o: openmp/%.c $(wildcard openmp/*.h) common/output_path.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPENMP_CFLAGS) -Iopenmp -c $< -o $@

$(BUILD_DIR)/mpi/%.o: mpi/%.c $(wildcard mpi/*.h) common/output_path.h
	@mkdir -p $(@D)
	$(MPICC) $(CPPFLAGS) $(CFLAGS) -Impi -c $< -o $@

$(BUILD_DIR)/cuda/%.o: cuda/%.c $(wildcard cuda/*.h) common/output_path.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -Icuda -c $< -o $@

$(BUILD_DIR)/cuda/%.o: cuda/%.cu $(wildcard cuda/*.h)
	@mkdir -p $(@D)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -Icuda -c $< -o $@

$(BUILD_DIR)/airfoil_serial: $(SERIAL_OBJ) $(BUILD_DIR)/serial/airfoil.o
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/airfoil_serial_opt: $(SERIAL_OBJ) $(BUILD_DIR)/serial/airfoil_opt.o
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/airfoil_openmp: $(OMP_OBJ) $(BUILD_DIR)/openmp/airfoil_omp.o
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(OPENMP_LDFLAGS) $(LDLIBS) -o $@

$(BUILD_DIR)/airfoil_openmp_opt: $(OMP_OBJ) $(BUILD_DIR)/openmp/airfoil_omp_optimized.o
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(OPENMP_LDFLAGS) $(LDLIBS) -o $@

$(BUILD_DIR)/airfoil_mpi: $(MPI_OBJ) $(BUILD_DIR)/mpi/airfoil_mpi.o
	$(MPICC) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/airfoil_cuda: $(CUDA_OBJ)
	$(NVCC) $(LDFLAGS) $(NVCCFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/airfoil_reference: $(wildcard reference/serial/*.c) $(wildcard reference/serial/*.h)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -include string.h -Ireference/serial $(filter-out reference/serial/airfoil_opt.c,$(filter %.c,$^)) $(LDFLAGS) $(LDLIBS) -o $@

check: cpu
	$(PYTHON) scripts/check_historical.py
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) scripts/validate.py --build-dir $(BUILD_DIR)

check-historical:
	$(PYTHON) scripts/check_historical.py

validate: check

info:
	$(PYTHON) scripts/build_info.py --build-dir $(BUILD_DIR) --cc="$(CC)" --mpicc="$(MPICC)" --nvcc="$(NVCC)" --cflags="$(CFLAGS)" --omp-cflags="$(OPENMP_CFLAGS)" --omp-ldflags="$(OPENMP_LDFLAGS)" --nvccflags="$(NVCCFLAGS)"

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo 'make demo       Run a verified CPU example and generate an offline result page'
	@echo 'make cpu        Build serial and OpenMP variants (default)'
	@echo 'make mpi        Build MPI variant (requires mpicc)'
	@echo 'make cuda       Build CUDA variant (requires nvcc; CUDA_ARCH=75 by default)'
	@echo 'make reference  Build archived serial reference (known legacy issues)'
	@echo 'make check      Run Python tests and numerical backend validation'
	@echo 'make check-historical  Verify archived logs, hashes and timing summaries'
	@echo 'Override CC, MPICC, CFLAGS, BUILD_DIR, CUDA_ARCH and PYTHON as needed.'
