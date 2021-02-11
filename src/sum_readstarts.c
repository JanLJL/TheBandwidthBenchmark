#include <timing.h>
#include <omp.h>
#include <immintrin.h>

double sum1start(
        double * restrict a,
        int N
        )
{
    double S, E;

    S = getTimeStamp();
    #pragma omp parallel
    {
        double sum = 0.0;
        for (int i=0; i<N; i++) {
            sum += a[i];
        //for (int i=0; i<N; i+=8) {
        //    __m512i lvec = _mm512_stream_load_si512 (&a[i]);
        //    sum += _mm512_reduce_add_epi64(lvec);
        }

        /* make the compiler think this makes actually sense */
        #pragma omp single
        a[10] = sum;
    } // omp parallel
    E = getTimeStamp();

    return E-S;
}

double sumMstart(
        double * restrict a,
        int N
        )
{
    double S, E;
    //int offset_mul = N / omp_get_max_threads();
    int offset_mul = 8;

    S = getTimeStamp();
    #pragma omp parallel
    {
        int offset = omp_get_thread_num()*offset_mul;
        double sum = 0.0;
        // go from thread-dependend start to end of array
        for (int i=offset; i<N; i++) {
            sum += a[i];
        }
        // go from start of array to thread-dependend end
        for (int i=0; i<offset; i++) {
            sum += a[i];
        }

        /* make the compiler think this makes actually sense */
        #pragma omp single
        a[10] = sum;
    } // omp parallel
    E = getTimeStamp();

    return E-S;
}
