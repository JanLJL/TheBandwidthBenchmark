# Supported: GCC, CLANG, ICC
TAG ?= GCC
ENABLE_OPENMP ?= false

#Feature options
OPTIONS  =  -DSIZE=40000000ull
OPTIONS +=  -DNTIMES=10
OPTIONS +=  -DARRAY_ALIGNMENT=64
#OPTIONS +=  -DVERBOSE_AFFINITY
#OPTIONS +=  -DVERBOSE_DATASIZE
#OPTIONS +=  -DVERBOSE_TIMER
