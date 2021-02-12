#include <timing.h>
#include <omp.h>
#include <stdio.h>
#include <immintrin.h>

double sumBdepOnVal(
        uint64_t * restrict a,
        double * restrict b,
        int N
        )
{
    double S, E;
    const int nthreads = omp_get_max_threads();
    __m512i _nthreads = _mm512_set_epi64(nthreads, nthreads, nthreads, nthreads, 
                                                nthreads, nthreads, nthreads, nthreads); 

    S = getTimeStamp();
    #pragma omp parallel
    {
        uint64_t tid = omp_get_thread_num();
        __m512i _tid = _mm512_set_epi64(tid, tid, tid, tid, tid, tid, tid, tid);
        //__mmask8 bmaskAll1 = _cvtu32_mask8(255);
        double sum = 0.0;
        double sum0 = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;
        double sum3 = 0.0;
        double sum4 = 0.0;
        double sum5 = 0.0;
        double sum6 = 0.0;
        double sum7 = 0.0;

        double val;
        #pragma unroll(4)
        for (int i=0; i<N; i+=64) {
            //printf("%d-%d: %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu\n", i, i+7, a[i], a[i+1], a[i+2], a[i+3], a[i+4], a[i+5], a[i+6], a[i+7]);
            /////////////////////////
            // INTRINSICS (+= 8)
            /////////////////////////
            //__m512i lvec = _mm512_load_epi64 (&a[i]);             // val = a[i] (-- a[i+7])
            //__m512i mod = _mm512_rem_epu64 (lvec, _nthreads);     // [(val % nthreads)         |
            //__mmask8 bmask = _mm512_cmpeq_epu64_mask (mod, _tid); // [_                == tid _|
            //__m512d bvec = _mm512_maskz_load_pd(bmask, &b[i]);    //         b[i] (-- b[i+7])
            //sum += _mm512_reduce_add_pd(bvec);                    //  sum +=
            /////////////////////////

            /////////////////////////
            // INTRINSICS (+= 64)
            /////////////////////////
            //__m512i lvec0 = _mm512_load_epi64 (&a[i]);
            //__m512i lvec1 = _mm512_load_epi64 (&a[i+8]);
            //__m512i mod0 = _mm512_rem_epu64 (lvec0, _nthreads);
            //__mmask8 bmask0 = _mm512_cmpeq_epu64_mask (mod0, _tid);
            //__m512d bvec0 = _mm512_maskz_load_pd(bmask0, &b[i]);
            //__m512i lvec2 = _mm512_load_epi64 (&a[i+16]);
            //__m512i mod1 = _mm512_rem_epu64 (lvec1, _nthreads);
            //__mmask8 bmask1 = _mm512_cmpeq_epu64_mask (mod1, _tid);
            //sum0 += _mm512_reduce_add_pd(bvec0);
            //__m512d bvec1 = _mm512_maskz_load_pd(bmask1, &b[i+8]);
            //__m512i lvec3 = _mm512_load_epi64 (&a[i+24]);
            //__m512i mod2 = _mm512_rem_epu64 (lvec2, _nthreads);
            //__mmask8 bmask2 = _mm512_cmpeq_epu64_mask (mod2, _tid);
            //sum1 += _mm512_reduce_add_pd(bvec1);
            //__m512i mod3 = _mm512_rem_epu64 (lvec3, _nthreads);
            //__m512d bvec2 = _mm512_maskz_load_pd(bmask2, &b[i+16]);
            //__mmask8 bmask3 = _mm512_cmpeq_epu64_mask (mod3, _tid);
            //__m512d bvec3 = _mm512_maskz_load_pd(bmask3, &b[i+24]);

            //__m512i lvec4 = _mm512_load_epi64 (&a[i+32]);
            //__m512i lvec5 = _mm512_load_epi64 (&a[i+40]);
            //sum2 += _mm512_reduce_add_pd(bvec2);
            //__m512i mod4 = _mm512_rem_epu64 (lvec4, _nthreads);
            //__mmask8 bmask4 = _mm512_cmpeq_epu64_mask (mod4, _tid);
            //sum3 += _mm512_reduce_add_pd(bvec3);
            //__m512d bvec4 = _mm512_maskz_load_pd(bmask4, &b[i+32]);
            //__m512i lvec6 = _mm512_load_epi64 (&a[i+48]);
            //__m512i mod5 = _mm512_rem_epu64 (lvec5, _nthreads);
            //__mmask8 bmask5 = _mm512_cmpeq_epu64_mask (mod5, _tid);
            //sum4 += _mm512_reduce_add_pd(bvec4);
            //__m512d bvec5 = _mm512_maskz_load_pd(bmask5, &b[i+40]);
            //__m512i lvec7 = _mm512_load_epi64 (&a[i+56]);
            //__m512i mod6 = _mm512_rem_epu64 (lvec6, _nthreads);
            //__mmask8 bmask6 = _mm512_cmpeq_epu64_mask (mod6, _tid);
            //sum5 += _mm512_reduce_add_pd(bvec5);
            //__m512i mod7 = _mm512_rem_epu64 (lvec7, _nthreads);
            //__m512d bvec6 = _mm512_maskz_load_pd(bmask6, &b[i+48]);
            //__mmask8 bmask7 = _mm512_cmpeq_epu64_mask (mod7, _tid);
            //__m512d bvec7 = _mm512_maskz_load_pd(bmask7, &b[i+56]);

            //sum6 += _mm512_reduce_add_pd(bvec6);
            //sum7 += _mm512_reduce_add_pd(bvec7);
            /////////////////////////

            /////////////////////////
            // INTRINSICS (+= 64)
            /////////////////////////
            _mm_prefetch(&b[i], _MM_HINT_T0);
            _mm_prefetch(&b[i+8], _MM_HINT_T0);
            _mm_prefetch(&b[i+16], _MM_HINT_T0);
            _mm_prefetch(&b[i+24], _MM_HINT_T0);
            _mm_prefetch(&b[i+32], _MM_HINT_T0);
            _mm_prefetch(&b[i+40], _MM_HINT_T0);
            _mm_prefetch(&b[i+48], _MM_HINT_T0);
            _mm_prefetch(&b[i+54], _MM_HINT_T0);
            __m512i lvec0 = _mm512_load_epi64 (&a[i]);
            __m512i lvec1 = _mm512_load_epi64 (&a[i+8]);
            __m512i mod0 = _mm512_rem_epu64 (lvec0, _nthreads);
            __mmask8 bmask0 = _mm512_cmpeq_epu64_mask (mod0, _tid);
            __m512d bvec0 = _mm512_maskz_load_pd(bmask0, &b[i]);
            __m512i lvec2 = _mm512_load_epi64 (&a[i+16]);
            __m512i mod1 = _mm512_rem_epu64 (lvec1, _nthreads);
            __mmask8 bmask1 = _mm512_cmpeq_epu64_mask (mod1, _tid);
            sum0 += _mm512_reduce_add_pd(bvec0);
            __m512d bvec1 = _mm512_maskz_load_pd(bmask1, &b[i+8]);
            __m512i lvec3 = _mm512_load_epi64 (&a[i+24]);
            __m512i mod2 = _mm512_rem_epu64 (lvec2, _nthreads);
            __mmask8 bmask2 = _mm512_cmpeq_epu64_mask (mod2, _tid);
            sum1 += _mm512_reduce_add_pd(bvec1);
            __m512i mod3 = _mm512_rem_epu64 (lvec3, _nthreads);
            __m512d bvec2 = _mm512_maskz_load_pd(bmask2, &b[i+16]);
            __mmask8 bmask3 = _mm512_cmpeq_epu64_mask (mod3, _tid);
            __m512d bvec3 = _mm512_maskz_load_pd(bmask3, &b[i+24]);

            __m512i lvec4 = _mm512_load_epi64 (&a[i+32]);
            __m512i lvec5 = _mm512_load_epi64 (&a[i+40]);
            sum2 += _mm512_reduce_add_pd(bvec2);
            __m512i mod4 = _mm512_rem_epu64 (lvec4, _nthreads);
            __mmask8 bmask4 = _mm512_cmpeq_epu64_mask (mod4, _tid);
            sum3 += _mm512_reduce_add_pd(bvec3);
            __m512d bvec4 = _mm512_maskz_load_pd(bmask4, &b[i+32]);
            __m512i lvec6 = _mm512_load_epi64 (&a[i+48]);
            __m512i mod5 = _mm512_rem_epu64 (lvec5, _nthreads);
            __mmask8 bmask5 = _mm512_cmpeq_epu64_mask (mod5, _tid);
            sum4 += _mm512_reduce_add_pd(bvec4);
            __m512d bvec5 = _mm512_maskz_load_pd(bmask5, &b[i+40]);
            __m512i lvec7 = _mm512_load_epi64 (&a[i+56]);
            __m512i mod6 = _mm512_rem_epu64 (lvec6, _nthreads);
            __mmask8 bmask6 = _mm512_cmpeq_epu64_mask (mod6, _tid);
            sum5 += _mm512_reduce_add_pd(bvec5);
            __m512i mod7 = _mm512_rem_epu64 (lvec7, _nthreads);
            __m512d bvec6 = _mm512_maskz_load_pd(bmask6, &b[i+48]);
            __mmask8 bmask7 = _mm512_cmpeq_epu64_mask (mod7, _tid);
            __m512d bvec7 = _mm512_maskz_load_pd(bmask7, &b[i+56]);

            sum6 += _mm512_reduce_add_pd(bvec6);
            sum7 += _mm512_reduce_add_pd(bvec7);
            /////////////////////////
            /////////////////////////
            // HIGH LEVEL CODE
            /////////////////////////
            //val = a[i];
            //if (val % nthreads == tid) {
            //    printf("%d: val= %lld\n", tid, val);
            //    sum += b[i];
            //}
            /////////////////////////
        }

        /* make the compiler think this makes actually sense */
        #pragma omp single
        a[10] = (uint64_t) sum0+sum1+sum2+sum3+sum4+sum5+sum6+sum7;
        //#pragma omp master
        //printf("sum=%f\n", sum);
    } // omp parallel
    E = getTimeStamp();

    return E-S;
}

double sumBdepOnVal64(
        uint64_t * restrict a,
        double * restrict b,
        int N
        )
{
    double S, E;
    const int nthreads = omp_get_max_threads();

    S = getTimeStamp();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double sum = 0.0;
        uint64_t val;
        for (int i=0; i<N; i+=8) {
            val = a[i];
            if (val % nthreads == tid) {
                //printf("%d: val= %lld\n", tid, val);
                #pragma omp simd
                for (int k=0; k<64; ++k) {
                    sum += b[i+k];
                }
            }
        }

        /* make the compiler think this makes actually sense */
        #pragma omp single
        a[10] = (uint64_t) sum;
        //#pragma omp master
        //printf("sum=%f\n", sum);
    } // omp parallel
    E = getTimeStamp();

    return E-S;
}
