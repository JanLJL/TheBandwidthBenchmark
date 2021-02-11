# Supported: GCC, CLANG, ICC
TAG ?= ICC
ENABLE_OPENMP ?= true
ENABLE_LIKWID ?= true

#Feature options
OPTIONS  =  -DSIZE=120000000ull
OPTIONS +=  -DNTIMES=10
OPTIONS +=  -DARRAY_ALIGNMENT=64
#OPTIONS +=  -DVERBOSE_AFFINITY
#OPTIONS +=  -DVERBOSE_DATASIZE
#OPTIONS +=  -DVERBOSE_TIMER
