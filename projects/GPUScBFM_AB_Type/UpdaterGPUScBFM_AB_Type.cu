/*
 * UpdaterGPUScBFM_AB_Type.cpp
 *
 *  Created on: 27.07.2017
 *      Author: Ron Dockhorn
 */


#include <cstdio>                           // printf
#include <cstdlib>                          // exit
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <stdint.h>
#include <sstream>

#include "UpdaterGPUScBFM_AB_Type.h"

#define DEBUG_UPDATERGPUSCBFM_AB_TYPE 100


/* why 512??? Because 512==8^3 ??? but that would mean 8 possible values instead of
 * -4 to +4 which I saw being used ... */
__device__ __constant__ bool dpForbiddenBonds [512]; //false-allowed; true-forbidden

/* ??? meaning
 * DXTable_d = { -1,1,0,0,0,0 }
 * DYTable_d = { 0,0,-1,1,0,0 }
 * DZTable_d = { 0,0,0,0,-1,1 }
 */
__device__ __constant__ intCUDA DXTable_d [6]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
__device__ __constant__ intCUDA DYTable_d [6]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
__device__ __constant__ intCUDA DZTable_d [6]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z

__device__ __constant__ uint32_t nMonomersSpeciesA_d;  // Nr of Monomer Species A
__device__ __constant__ uint32_t nMonomersSpeciesB_d;  // Nr of Monomer Species B

__device__ __constant__ uint32_t LATTICE_X_d    ;  // mLattice size in X
__device__ __constant__ uint32_t LATTICE_Y_d    ;  // mLattice size in Y
__device__ __constant__ uint32_t LATTICE_Z_d    ;  // mLattice size in Z
/* will this really bring performance improvement? At least constant cache
 * might be as fast as register access when all threads in a warp access the
 * the same constant */
__device__ __constant__ uint32_t LATTICE_XM1_d  ;  // mLattice size in X-1
__device__ __constant__ uint32_t LATTICE_YM1_d  ;  // mLattice size in Y-1
__device__ __constant__ uint32_t LATTICE_ZM1_d  ;  // mLattice size in Z-1

__device__ __constant__ uint32_t LATTICE_XPRO_d ;  // mLattice shift in X
__device__ __constant__ uint32_t LATTICE_PROXY_d;  // mLattice shift in X*Y

/* Since CUDA 5.5 (~2014) there do exist texture objects which are much
 * easier and can actually be used as kernel arguments!
 * @see https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/
 * "What is not commonly known is that each outstanding texture reference that
 *  is bound when a kernel is launched incurs added launch latency—up to 0.5 μs
 *  per texture reference. This launch overhead persists even if the outstanding
 *  bound textures are not even referenced by the kernel. Again, using texture
 *  objects instead of texture references completely removes this overhead."
 * -> wow !!!
 */
texture< uint8_t, cudaTextureType1D, cudaReadModeElementType > texmLatticeRefOut;
texture< uint8_t, cudaTextureType1D, cudaReadModeElementType > texmLatticeTmpRef;
texture< intCUDA, cudaTextureType1D, cudaReadModeElementType > texPolymerAndMonomerIsEvenAndOnXRef;
/* These are arrays containing the monomer indices for the respective
 * species (sorted ascending). E.g. for AABABBA this would be:
 * texSpeciesIndicesA = { 0,1,3,6 }
 * texSpeciesIndicesB = { 1,4,5 }
 */
cudaTextureObject_t texSpeciesIndicesA = 0;
cudaTextureObject_t texSpeciesIndicesB = 0;
texture< int32_t, cudaTextureType1D, cudaReadModeElementType > texMonomersSpezies_A_ThreadIdx;
texture< int32_t, cudaTextureType1D, cudaReadModeElementType > texMonomersSpezies_B_ThreadIdx;

uint32_t nMonomersSpeciesA;
uint32_t nMonomersSpeciesB;

__device__ uint32_t hash(uint32_t a)
{
    /* https://web.archive.org/web/20120626084524/http://www.concentric.net:80/~ttwang/tech/inthash.htm
     * Note that before this 2007-03 version there were no magic numbers.
     * This hash function doesn't seem to be published.
     * He writes himself that this shouldn't really be used for PRNGs ???
     * @todo E.g. check random distribution of randomly drawn directions are
     *       they rouhgly even?
     * The 'hash' or at least an older version of it can even be inverted !!!
     * http://c42f.github.io/2015/09/21/inverting-32-bit-wang-hash.html
     * Somehow this also gets attibuted to Robert Jenkins?
     * https://gist.github.com/badboy/6267743
     * -> http://www.burtleburtle.net/bob/hash/doobs.html
     *    http://burtleburtle.net/bob/hash/integer.html
     */
    a = ( a + 0x7ed55d16 ) + ( a << 12 );
    a = ( a ^ 0xc761c23c ) ^ ( a >> 19 );
    a = ( a + 0x165667b1 ) + ( a << 5  );
    a = ( a + 0xd3a2646c ) ^ ( a << 9  );
    a = ( a + 0xfd7046c5 ) + ( a << 3  );
    a = ( a ^ 0xb55a4f09 ) ^ ( a >> 16 );
    return a;
}

__device__ uintCUDA IdxBondArray_d
(
    intCUDA const x,
    intCUDA const y,
    intCUDA const z
)
{
    return   ( x & 7 ) +
           ( ( y & 7 ) << 3 ) +
           ( ( z & 7 ) << 6 );
}


__device__ inline bool checkLattice
(
    intCUDA  const x0,
    intCUDA  const y0,
    intCUDA  const z0,
    intCUDA  const dx,
    intCUDA  const dy,
    intCUDA  const dz,
    uintCUDA const axis
)
{
    uint8_t test = 0;
    /* positions after movement. Why 2 times dx ??? */
    uint32_t const x1 = ( x0 + dx + dx ) & LATTICE_XM1_d;
    uint32_t const y1 = ( y0 + dy + dy ) & LATTICE_YM1_d;
    uint32_t const z1 = ( z0 + dz + dz ) & LATTICE_ZM1_d;
    uint32_t const x0Abs  = ( x0     ) & LATTICE_XM1_d;
    uint32_t const x0PDX  = ( x0 + 1 ) & LATTICE_XM1_d;
    uint32_t const x0MDX  = ( x0 - 1 ) & LATTICE_XM1_d;
    uint32_t const y0Abs  = ( y0     ) & LATTICE_YM1_d;
    uint32_t const y0PDY  = ( y0 + 1 ) & LATTICE_YM1_d;
    uint32_t const y0MDY  = ( y0 - 1 ) & LATTICE_YM1_d;
    uint32_t const z0Abs  = ( z0     ) & LATTICE_ZM1_d;
    uint32_t const z0PDZ  = ( z0 + 1 ) & LATTICE_ZM1_d;
    uint32_t const z0MDZ  = ( z0 - 1 ) & LATTICE_ZM1_d;

    switch ( axis )
    {
        case 0: //-+x
            test = tex1Dfetch( texmLatticeRefOut, x1 + (y0Abs << LATTICE_XPRO_d) + (z0Abs << LATTICE_PROXY_d ) ) |
                   tex1Dfetch( texmLatticeRefOut, x1 + (y0PDY << LATTICE_XPRO_d) + (z0Abs << LATTICE_PROXY_d ) ) |
                   tex1Dfetch( texmLatticeRefOut, x1 + (y0MDY << LATTICE_XPRO_d) + (z0Abs << LATTICE_PROXY_d ) ) |
                   tex1Dfetch( texmLatticeRefOut, x1 + (y0Abs << LATTICE_XPRO_d) + (z0PDZ << LATTICE_PROXY_d ) ) |
                   tex1Dfetch( texmLatticeRefOut, x1 + (y0Abs << LATTICE_XPRO_d) + (z0MDZ << LATTICE_PROXY_d ) ) |
                   tex1Dfetch( texmLatticeRefOut, x1 + (y0MDY << LATTICE_XPRO_d) + (z0MDZ << LATTICE_PROXY_d ) ) |
                   tex1Dfetch( texmLatticeRefOut, x1 + (y0PDY << LATTICE_XPRO_d) + (z0MDZ << LATTICE_PROXY_d ) ) |
                   tex1Dfetch( texmLatticeRefOut, x1 + (y0MDY << LATTICE_XPRO_d) + (z0PDZ << LATTICE_PROXY_d ) ) |
                   tex1Dfetch( texmLatticeRefOut, x1 + (y0PDY << LATTICE_XPRO_d) + (z0PDZ << LATTICE_PROXY_d ) );
            break;

        case 1: //-+y
            test = tex1Dfetch( texmLatticeRefOut, x0MDX + (y1 << LATTICE_XPRO_d) + (z0MDZ << LATTICE_PROXY_d ) ) |
                   tex1Dfetch( texmLatticeRefOut, x0Abs + (y1 << LATTICE_XPRO_d) + (z0MDZ << LATTICE_PROXY_d ) ) |
                   tex1Dfetch( texmLatticeRefOut, x0PDX + (y1 << LATTICE_XPRO_d) + (z0MDZ << LATTICE_PROXY_d ) ) |
                   tex1Dfetch( texmLatticeRefOut, x0MDX + (y1 << LATTICE_XPRO_d) + (z0Abs << LATTICE_PROXY_d ) ) |
                   tex1Dfetch( texmLatticeRefOut, x0Abs + (y1 << LATTICE_XPRO_d) + (z0Abs << LATTICE_PROXY_d ) ) |
                   tex1Dfetch( texmLatticeRefOut, x0PDX + (y1 << LATTICE_XPRO_d) + (z0Abs << LATTICE_PROXY_d ) ) |
                   tex1Dfetch( texmLatticeRefOut, x0MDX + (y1 << LATTICE_XPRO_d) + (z0PDZ << LATTICE_PROXY_d ) ) |
                   tex1Dfetch( texmLatticeRefOut, x0Abs + (y1 << LATTICE_XPRO_d) + (z0PDZ << LATTICE_PROXY_d ) ) |
                   tex1Dfetch( texmLatticeRefOut, x0PDX + (y1 << LATTICE_XPRO_d) + (z0PDZ << LATTICE_PROXY_d ) );
            break;

        case 2: //-+z
            test = tex1Dfetch( texmLatticeRefOut, x0MDX + (y0MDY << LATTICE_XPRO_d) + (z1 << LATTICE_PROXY_d ) ) |
                   tex1Dfetch( texmLatticeRefOut, x0Abs + (y0MDY << LATTICE_XPRO_d) + (z1 << LATTICE_PROXY_d ) ) |
                   tex1Dfetch( texmLatticeRefOut, x0PDX + (y0MDY << LATTICE_XPRO_d) + (z1 << LATTICE_PROXY_d ) ) |
                   tex1Dfetch( texmLatticeRefOut, x0MDX + (y0Abs << LATTICE_XPRO_d) + (z1 << LATTICE_PROXY_d ) ) |
                   tex1Dfetch( texmLatticeRefOut, x0Abs + (y0Abs << LATTICE_XPRO_d) + (z1 << LATTICE_PROXY_d ) ) |
                   tex1Dfetch( texmLatticeRefOut, x0PDX + (y0Abs << LATTICE_XPRO_d) + (z1 << LATTICE_PROXY_d ) ) |
                   tex1Dfetch( texmLatticeRefOut, x0MDX + (y0PDY << LATTICE_XPRO_d) + (z1 << LATTICE_PROXY_d ) ) |
                   tex1Dfetch( texmLatticeRefOut, x0Abs + (y0PDY << LATTICE_XPRO_d) + (z1 << LATTICE_PROXY_d ) ) |
                   tex1Dfetch( texmLatticeRefOut, x0PDX + (y0PDY << LATTICE_XPRO_d) + (z1 << LATTICE_PROXY_d ) );
            break;
    }
    return test;
}


/**
 * @param[in] rn a random number used as a kind of seed for the RNG
 * @param[in] nMonomers number of max. monomers to work on, this is for
 *            filtering out excessive threads and was prior a __constant__
 *            But it is only used one(!) time in the kernel so the caching
 *            of constant memory might not even be used.
 *            @see https://web.archive.org/web/20140612185804/http://www.pixel.io/blog/2013/5/9/kernel-arguments-vs-__constant__-variables.html
 *            -> Kernel arguments are even put into constant memory it seems:
 *            @see "Section E.2.5.2 Function Parameters" in the "CUDA 5.5 C Programming Guide"
 *            __global__ function parameters are passed to the device:
 *             - via shared memory and are limited to 256 bytes on devices of compute capability 1.x,
 *             - via constant memory and are limited to 4 KB on devices of compute capability 2.x and higher.
 *            __device__ and __global__ functions cannot have a variable number of arguments.
 */
__global__ void kernelSimulationScBFMCheckSpezies
(
    intCUDA  * const mPolymerSystem_d,
    uint8_t  * const mLatticeTmp_d   ,
    MonoInfo * const MonoInfo_d      ,
    cudaTextureObject_t const texSpeciesIndices,
    uint32_t   const nMonomers       ,
    uint32_t   const rn
)
{
    int linId = blockIdx.x * blockDim.x + threadIdx.x;

    /* might be more readable to just return if the thread is masked ???
     * if ( ! ( linId < nMonomersSpeciesA_d ) )
     *     return;
     * I think it only works on newer CUDA versions ??? else the whole warp
     * might quit???
     */
    if ( linId < nMonomers )
    {
        //select random monomer ??? I don't see why this is random? Is texMonomersSpezies_A_ThreadIdx randomized?
        uint32_t const randomMonomer = tex1Dfetch< uint32_t >( texSpeciesIndices, linId );

        //select random direction. Own implementation of an rng :S? But I think it at least# was initialized using the LeMonADE RNG ...
        uintCUDA const random_int = hash(hash(linId) ^ rn) % 6;

        intCUDA const x0           = tex1Dfetch( texPolymerAndMonomerIsEvenAndOnXRef, 4*randomMonomer+0 );
        intCUDA const y0           = tex1Dfetch( texPolymerAndMonomerIsEvenAndOnXRef, 4*randomMonomer+1 );
        intCUDA const z0           = tex1Dfetch( texPolymerAndMonomerIsEvenAndOnXRef, 4*randomMonomer+2 );
        intCUDA const MonoProperty = tex1Dfetch( texPolymerAndMonomerIsEvenAndOnXRef, 4*randomMonomer+3 );

         //select random direction. !!! That table is kinda magic. there might be a better way ... E.g. using bitmasking. Also, what is with 0 in one direction ??? There is no way to e.g. get (0,1,-1) ... ???
         //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
        intCUDA const dx = DXTable_d[ random_int ];
        intCUDA const dy = DYTable_d[ random_int ];
        intCUDA const dz = DZTable_d[ random_int ];

#ifdef NONPERIODICITY
       /* check whether the new location of the particle would be inside the box
        * if the box is not periodic, if not, then don't move the particle */
        if ( ! ( 0 <= x0 + dx && x0 + dx < LATTICE_XM1_d &&
                 0 <= y0 + dy && y0 + dy < LATTICE_YM1_d &&
                 0 <= z0 + dz && z0 + dz < LATTICE_ZM1_d ) )
        {
            return;
        }
#endif
        /* ??? why 224 = 0xE0 = 0b1110 0000 */
        const unsigned nextNeigborSize = ( MonoProperty & 224 ) >> 5;
        for ( unsigned u = 0; u < nextNeigborSize; ++u )
        {
            intCUDA const nN_X=tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*MonoInfo_d[randomMonomer].bondsMonomerIdx[u]  );
            intCUDA const nN_Y=tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*MonoInfo_d[randomMonomer].bondsMonomerIdx[u]+1);
            intCUDA const nN_Z=tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*MonoInfo_d[randomMonomer].bondsMonomerIdx[u]+2);

            //dpForbiddenBonds [512]; //false-allowed; true-forbidden
            if ( dpForbiddenBonds[IdxBondArray_d(nN_X-x0-dx, nN_Y-y0-dy, nN_Z-z0-dz)] )
                return;
        }

        //check the lattice
        if ( checkLattice( x0, y0, z0, dx, dy, dz, random_int >> 1 ) )
            return;

        // everything fits -> perform the move - add the information
        // possible move
        /* ??? can I simply & LATTICE_XM1_d ? this looks like something like
         * ( x0+dx ) % xmax is trying to be achieved. Using bitmasking for that
         * is only possible if LATTICE_X_d is a power of two ... */
        mPolymerSystem_d[4*randomMonomer+3] = MonoProperty | ((random_int<<2)+1);
        mLatticeTmp_d[  ( ( x0 + dx ) & LATTICE_XM1_d ) +
                      ( ( ( y0 + dy ) & LATTICE_YM1_d ) << LATTICE_XPRO_d ) +
                      ( ( ( z0 + dz ) & LATTICE_ZM1_d ) << LATTICE_PROXY_d ) ] = 1;
    }
}

/**
 * !!! has many identical code parts as runSimulationScBFMCheckSpeziesA_gpu
 */
__global__ void runSimulationScBFMPerformSpeziesA_gpu
(
    intCUDA * mPolymerSystem_d,
    uint8_t * mLattice_d
)
{
    int const idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idxA < nMonomersSpeciesA_d )
    {
        //select random monomer. Again why is this random ... ???
        uint32_t const randomMonomer = tex1Dfetch( texMonomersSpezies_A_ThreadIdx, idxA );
        intCUDA  const MonoProperty = tex1Dfetch( texPolymerAndMonomerIsEvenAndOnXRef, 4 * randomMonomer + 3 );

        if ( MonoProperty & 1 != 0 )    // possible move
        {
            intCUDA const x0 = tex1Dfetch( texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+0 );
            intCUDA const y0 = tex1Dfetch( texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+1 );
            intCUDA const z0 = tex1Dfetch( texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+2 );

            //select random direction
            uintCUDA const random_int = (MonoProperty&28)>>2;

            //select random direction
            //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z

            /*
            tmp_DXTable[0]=-1; tmp_DXTable[1]= 1; tmp_DXTable[2]= 0; tmp_DXTable[3]= 0; tmp_DXTable[4]= 0; tmp_DXTable[5]= 0;
            tmp_DYTable[0]= 0; tmp_DYTable[1]= 0; tmp_DYTable[2]=-1; tmp_DYTable[3]= 1; tmp_DYTable[4]= 0; tmp_DYTable[5]= 0;
            tmp_DZTable[0]= 0; tmp_DZTable[1]= 0; tmp_DZTable[2]= 0; tmp_DZTable[3]= 0; tmp_DZTable[4]=-1; tmp_DZTable[5]= 1;
            */
            intCUDA const dx = DXTable_d[ random_int ];
            intCUDA const dy = DYTable_d[ random_int ];
            intCUDA const dz = DZTable_d[ random_int ];

            //check the lattice
            uint8_t test = 0;

            uint32_t const x1 = ( x0 + dx + dx ) & LATTICE_XM1_d;
            uint32_t const y1 = ( y0 + dy + dy ) & LATTICE_YM1_d;
            uint32_t const z1 = ( z0 + dz + dz ) & LATTICE_ZM1_d;

            uint32_t const x0Abs  = ( x0     ) & LATTICE_XM1_d;
            uint32_t const x0PDX  = ( x0 + 1 ) & LATTICE_XM1_d;
            uint32_t const x0MDX  = ( x0 - 1 ) & LATTICE_XM1_d;
            uint32_t const y0Abs  = ( y0     ) & LATTICE_YM1_d;
            uint32_t const y0PDY  = ( y0 + 1 ) & LATTICE_YM1_d;
            uint32_t const y0MDY  = ( y0 - 1 ) & LATTICE_YM1_d;
            uint32_t const z0Abs  = ( z0     ) & LATTICE_ZM1_d;
            uint32_t const z0PDZ  = ( z0 + 1 ) & LATTICE_ZM1_d;
            uint32_t const z0MDZ  = ( z0 - 1 ) & LATTICE_ZM1_d;

            switch ( random_int >> 1 )
            {
                case 0: //-+x
                    test = tex1Dfetch(texmLatticeTmpRef, x1 + (y0Abs << LATTICE_XPRO_d) + (z0Abs << LATTICE_PROXY_d) ) |
                           tex1Dfetch(texmLatticeTmpRef, x1 + (y0PDY << LATTICE_XPRO_d) + (z0Abs << LATTICE_PROXY_d) ) |
                           tex1Dfetch(texmLatticeTmpRef, x1 + (y0MDY << LATTICE_XPRO_d) + (z0Abs << LATTICE_PROXY_d) ) |
                           tex1Dfetch(texmLatticeTmpRef, x1 + (y0Abs << LATTICE_XPRO_d) + (z0PDZ << LATTICE_PROXY_d) ) |
                           tex1Dfetch(texmLatticeTmpRef, x1 + (y0Abs << LATTICE_XPRO_d) + (z0MDZ << LATTICE_PROXY_d) ) |
                           tex1Dfetch(texmLatticeTmpRef, x1 + (y0MDY << LATTICE_XPRO_d) + (z0MDZ << LATTICE_PROXY_d) ) |
                           tex1Dfetch(texmLatticeTmpRef, x1 + (y0PDY << LATTICE_XPRO_d) + (z0MDZ << LATTICE_PROXY_d) ) |
                           tex1Dfetch(texmLatticeTmpRef, x1 + (y0MDY << LATTICE_XPRO_d) + (z0PDZ << LATTICE_PROXY_d) ) |
                           tex1Dfetch(texmLatticeTmpRef, x1 + (y0PDY << LATTICE_XPRO_d) + (z0PDZ << LATTICE_PROXY_d) );
                    break;
                case 1: //-+y
                    test = tex1Dfetch(texmLatticeTmpRef, x0MDX + (y1 << LATTICE_XPRO_d) + (z0MDZ << LATTICE_PROXY_d) ) |
                           tex1Dfetch(texmLatticeTmpRef, x0Abs + (y1 << LATTICE_XPRO_d) + (z0MDZ << LATTICE_PROXY_d) ) |
                           tex1Dfetch(texmLatticeTmpRef, x0PDX + (y1 << LATTICE_XPRO_d) + (z0MDZ << LATTICE_PROXY_d) ) |
                           tex1Dfetch(texmLatticeTmpRef, x0MDX + (y1 << LATTICE_XPRO_d) + (z0Abs << LATTICE_PROXY_d) ) |
                           tex1Dfetch(texmLatticeTmpRef, x0Abs + (y1 << LATTICE_XPRO_d) + (z0Abs << LATTICE_PROXY_d) ) |
                           tex1Dfetch(texmLatticeTmpRef, x0PDX + (y1 << LATTICE_XPRO_d) + (z0Abs << LATTICE_PROXY_d) ) |
                           tex1Dfetch(texmLatticeTmpRef, x0MDX + (y1 << LATTICE_XPRO_d) + (z0PDZ << LATTICE_PROXY_d) ) |
                           tex1Dfetch(texmLatticeTmpRef, x0Abs + (y1 << LATTICE_XPRO_d) + (z0PDZ << LATTICE_PROXY_d) ) |
                           tex1Dfetch(texmLatticeTmpRef, x0PDX + (y1 << LATTICE_XPRO_d) + (z0PDZ << LATTICE_PROXY_d) );

                    break;
                case 2: //-+z
                    test = tex1Dfetch(texmLatticeTmpRef, x0MDX + (y0MDY << LATTICE_XPRO_d) + (z1 << LATTICE_PROXY_d) ) |
                           tex1Dfetch(texmLatticeTmpRef, x0Abs + (y0MDY << LATTICE_XPRO_d) + (z1 << LATTICE_PROXY_d) ) |
                           tex1Dfetch(texmLatticeTmpRef, x0PDX + (y0MDY << LATTICE_XPRO_d) + (z1 << LATTICE_PROXY_d) ) |
                           tex1Dfetch(texmLatticeTmpRef, x0MDX + (y0Abs << LATTICE_XPRO_d) + (z1 << LATTICE_PROXY_d) ) |
                           tex1Dfetch(texmLatticeTmpRef, x0Abs + (y0Abs << LATTICE_XPRO_d) + (z1 << LATTICE_PROXY_d) ) |
                           tex1Dfetch(texmLatticeTmpRef, x0PDX + (y0Abs << LATTICE_XPRO_d) + (z1 << LATTICE_PROXY_d) ) |
                           tex1Dfetch(texmLatticeTmpRef, x0MDX + (y0PDY << LATTICE_XPRO_d) + (z1 << LATTICE_PROXY_d) ) |
                           tex1Dfetch(texmLatticeTmpRef, x0Abs + (y0PDY << LATTICE_XPRO_d) + (z1 << LATTICE_PROXY_d) ) |
                           tex1Dfetch(texmLatticeTmpRef, x0PDX + (y0PDY << LATTICE_XPRO_d) + (z1 << LATTICE_PROXY_d) );

                        break;

              }

              if (test) return;


          // everything fits -> perform the move - add the information
            //mPolymerSystem_d[4*randomMonomer  ] = x0 +dx;
            //mPolymerSystem_d[4*randomMonomer+1] = y0 +dy;
            //mPolymerSystem_d[4*randomMonomer+2] = z0 +dz;
            mPolymerSystem_d[4*randomMonomer+3] = MonoProperty | 2; // indicating allowed move
            mLattice_d[((x0 + dx     )&LATTICE_XM1_d) + (((y0 + dy     )&LATTICE_YM1_d) << LATTICE_XPRO_d) + (((z0 + dz) & LATTICE_ZM1_d) << LATTICE_PROXY_d)]=1;


            mLattice_d[x0Abs + (y0Abs << LATTICE_XPRO_d) + (z0Abs << LATTICE_PROXY_d)]=0;

      }

    }
}

__global__ void runSimulationScBFMZeroArraySpeziesA_gpu(intCUDA *mPolymerSystem_d, uint8_t *mLatticeTmp_d) {


      int idxA=blockIdx.x*blockDim.x+threadIdx.x;

      if(idxA < nMonomersSpeciesA_d)
      {
            //select random monomer
          uint32_t const randomMonomer=tex1Dfetch(texMonomersSpezies_A_ThreadIdx,idxA);


          intCUDA const MonoProperty = tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+3);

      if(((MonoProperty&3) != 0))    //possible move
      {
          intCUDA const x0= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer);
          intCUDA const y0= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+1);
          intCUDA const z0= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+2);

          //select random direction
          uintCUDA const random_int = (MonoProperty&28)>>2;

          //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
          intCUDA const dx = DXTable_d[random_int];
          intCUDA const dy = DYTable_d[random_int];
          intCUDA const dz = DZTable_d[random_int];


          // possible move but not allowed
          if(((MonoProperty&3) == 1))
          {
              mLatticeTmp_d[((x0 + dx     )&LATTICE_XM1_d) + (((y0 + dy     )&LATTICE_YM1_d) << LATTICE_XPRO_d) + (((z0 + dz) & LATTICE_ZM1_d) << LATTICE_PROXY_d)]=0;

              mPolymerSystem_d[4*randomMonomer+3] = MonoProperty & MASK5BITS; // delete the first 5 bits
          }
          else //allowed move with all circumstance
          {
              mPolymerSystem_d[4*randomMonomer  ] = x0 +dx;
              mPolymerSystem_d[4*randomMonomer+1] = y0 +dy;
              mPolymerSystem_d[4*randomMonomer+2] = z0 +dz;
              mPolymerSystem_d[4*randomMonomer+3] = MonoProperty & MASK5BITS; // delete the first 5 bits

              mLatticeTmp_d[((x0 + dx     )&LATTICE_XM1_d) + (((y0 + dy     )&LATTICE_YM1_d) << LATTICE_XPRO_d) + (((z0 + dz) & LATTICE_ZM1_d) << LATTICE_PROXY_d)]=0;

              //mLatticeTmp_d[((x0      )&LATTICE_XM1_d) + (((y0      )&LATTICE_YM1_d) << LATTICE_XPRO_d) + (((z0 ) & LATTICE_ZM1_d) << LATTICE_PROXY_d)]=0;

          }
          // everything fits -> perform the move - add the information

      }

      }
}


__global__ void runSimulationScBFMPerformSpeziesB_gpu(intCUDA *mPolymerSystem_d, uint8_t *mLattice_d) {


      int idxB=blockIdx.x*blockDim.x+threadIdx.x;


      if(idxB < nMonomersSpeciesB_d)
      {
          //select random monomer
          uint32_t const randomMonomer=tex1Dfetch(texMonomersSpezies_B_ThreadIdx,idxB);

          intCUDA const MonoProperty = tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+3);


    if(((MonoProperty&1) != 0))    //possible move
      {
        intCUDA const xPosMono= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer);
        intCUDA const yPosMono= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+1);
        intCUDA const zPosMono= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+2);

          //select random direction
        uintCUDA const random_int = (MonoProperty&28)>>2;

         //select random direction
         //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z

        intCUDA const dx = DXTable_d[random_int];
        intCUDA const dy = DYTable_d[random_int];
        intCUDA const dz = DZTable_d[random_int];

        //check the lattice
        uint8_t test = 0;

        uint32_t const xPosMonoDXDX = ((xPosMono + dx + dx)&LATTICE_XM1_d);
        const  uint32_t yPosMonoDYDY = ((yPosMono + dy + dy)&LATTICE_YM1_d);
        const  uint32_t zPosMonoDZDZ = ((zPosMono + dz + dz)&LATTICE_ZM1_d);

        const  uint32_t xPosMonoAbs = ((xPosMono          )&LATTICE_XM1_d);
        const  uint32_t xPosMonoPDX = ((xPosMono + 1      )&LATTICE_XM1_d);
        const  uint32_t xPosMonoMDX = ((xPosMono - 1      )&LATTICE_XM1_d);

        const  uint32_t yPosMonoAbs = ((yPosMono          )&LATTICE_YM1_d);
        const  uint32_t yPosMonoPDY = ((yPosMono + 1      )&LATTICE_YM1_d);
        const  uint32_t yPosMonoMDY = ((yPosMono - 1      )&LATTICE_YM1_d);

        const  uint32_t zPosMonoAbs = ((zPosMono          )&LATTICE_ZM1_d);
        const  uint32_t zPosMonoPDZ = ((zPosMono + 1      )&LATTICE_ZM1_d);
        const  uint32_t zPosMonoMDZ = ((zPosMono - 1      )&LATTICE_ZM1_d);

          switch (random_int >> 1)
              {
              case 0: //-+x

                    test =  tex1Dfetch(texmLatticeTmpRef, xPosMonoDXDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
                              tex1Dfetch(texmLatticeTmpRef, xPosMonoDXDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
                              tex1Dfetch(texmLatticeTmpRef, xPosMonoDXDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
                              tex1Dfetch(texmLatticeTmpRef, xPosMonoDXDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) ) |
                              tex1Dfetch(texmLatticeTmpRef, xPosMonoDXDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
                              tex1Dfetch(texmLatticeTmpRef, xPosMonoDXDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
                              tex1Dfetch(texmLatticeTmpRef, xPosMonoDXDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
                              tex1Dfetch(texmLatticeTmpRef, xPosMonoDXDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) ) |
                              tex1Dfetch(texmLatticeTmpRef, xPosMonoDXDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) );

                        break;

              case 1: //-+y

                  test =  tex1Dfetch(texmLatticeTmpRef, xPosMonoMDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
                            tex1Dfetch(texmLatticeTmpRef, xPosMonoAbs + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
                            tex1Dfetch(texmLatticeTmpRef, xPosMonoPDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |

                            tex1Dfetch(texmLatticeTmpRef, xPosMonoMDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
                            tex1Dfetch(texmLatticeTmpRef, xPosMonoAbs + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
                            tex1Dfetch(texmLatticeTmpRef, xPosMonoPDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |

                            tex1Dfetch(texmLatticeTmpRef, xPosMonoMDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) ) |
                            tex1Dfetch(texmLatticeTmpRef, xPosMonoAbs + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) ) |
                            tex1Dfetch(texmLatticeTmpRef, xPosMonoPDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) );

                        break;

              case 2: //-+z

                    test =  tex1Dfetch(texmLatticeTmpRef, xPosMonoMDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
                                tex1Dfetch(texmLatticeTmpRef, xPosMonoAbs + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
                                tex1Dfetch(texmLatticeTmpRef, xPosMonoPDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |

                                tex1Dfetch(texmLatticeTmpRef, xPosMonoMDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
                                tex1Dfetch(texmLatticeTmpRef, xPosMonoAbs + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
                                tex1Dfetch(texmLatticeTmpRef, xPosMonoPDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |

                                tex1Dfetch(texmLatticeTmpRef, xPosMonoMDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
                                tex1Dfetch(texmLatticeTmpRef, xPosMonoAbs + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
                            tex1Dfetch(texmLatticeTmpRef, xPosMonoPDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) );

                        break;

              }

        if (test) return;


          // everything fits -> perform the move - add the information

            //mPolymerSystem_d[4*randomMonomer  ] = xPosMono +dx;
            //mPolymerSystem_d[4*randomMonomer+1] = yPosMono +dy;
            //mPolymerSystem_d[4*randomMonomer+2] = zPosMono +dz;
            mPolymerSystem_d[4*randomMonomer+3] = MonoProperty | 2; // indicating allowed move
            mLattice_d[((xPosMono + dx     )&LATTICE_XM1_d) + (((yPosMono + dy     )&LATTICE_YM1_d) << LATTICE_XPRO_d) + (((zPosMono + dz) & LATTICE_ZM1_d) << LATTICE_PROXY_d)]=1;


            mLattice_d[xPosMonoAbs + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d)]=0;

      }

    }
}

__global__ void runSimulationScBFMZeroArraySpeziesB_gpu(intCUDA *mPolymerSystem_d, uint8_t *mLatticeTmp_d)
{
      int idxB=blockIdx.x*blockDim.x+threadIdx.x;

      if(idxB < nMonomersSpeciesB_d)
      {
            //select random monomer
          uint32_t const randomMonomer=tex1Dfetch(texMonomersSpezies_B_ThreadIdx,idxB);


          intCUDA const MonoProperty = tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+3);

      if(((MonoProperty&3) != 0))    //possible move
      {
          intCUDA const xPosMono= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer);
          intCUDA const yPosMono= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+1);
          intCUDA const zPosMono= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+2);

          //select random direction
          uintCUDA const random_int = (tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+3)&28)>>2;

          //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
          intCUDA const dx = DXTable_d[random_int];
          intCUDA const dy = DYTable_d[random_int];
          intCUDA const dz = DZTable_d[random_int];



          if(((MonoProperty&3) == 1))
          {
              mLatticeTmp_d[((xPosMono + dx     )&LATTICE_XM1_d) + (((yPosMono + dy     )&LATTICE_YM1_d) << LATTICE_XPRO_d) + (((zPosMono + dz) & LATTICE_ZM1_d) << LATTICE_PROXY_d)]=0;

              mPolymerSystem_d[4*randomMonomer+3] = MonoProperty & MASK5BITS; // delete the first 5 bits

          }
          else
          {
              mPolymerSystem_d[4*randomMonomer  ] = xPosMono +dx;
              mPolymerSystem_d[4*randomMonomer+1] = yPosMono +dy;
              mPolymerSystem_d[4*randomMonomer+2] = zPosMono +dz;
              mPolymerSystem_d[4*randomMonomer+3] = MonoProperty & MASK5BITS; // delete the first 5 bits

              mLatticeTmp_d[((xPosMono + dx     )&LATTICE_XM1_d) + (((yPosMono + dy     )&LATTICE_YM1_d) << LATTICE_XPRO_d) + (((zPosMono + dz) & LATTICE_ZM1_d) << LATTICE_PROXY_d)]=0;

              //mLatticeTmp_d[((xPosMono      )&LATTICE_XM1_d) + (((yPosMono      )&LATTICE_YM1_d) << LATTICE_XPRO_d) + (((zPosMono ) & LATTICE_ZM1_d) << LATTICE_PROXY_d)]=0;

          }
          // everything fits -> perform the move - add the information
          //  mPolymerSystem_d[4*randomMonomer+3] = MonoProperty & MASK5BITS; // delete the first 5 bits
      }
      }
}


UpdaterGPUScBFM_AB_Type::~UpdaterGPUScBFM_AB_Type()
{
    std::cout << "[" << __FILENAME__ << "::~UpdaterGPUScBFM_AB_Type" << "] destructor" << std::endl;

    delete[] mLattice;
    delete[] mPolymerSystem;
    delete[] mAttributeSystem;
    for ( size_t i = 0; i < nAllMonomers; ++i )
        delete monosNNidx[i];
    delete monosNNidx;

}

void UpdaterGPUScBFM_AB_Type::initialize( int iGpuToUse )
{
    /**** Print some GPU information ****/
    cudaDeviceProp prop;

    int nGpus;
    CUDA_CHECK( cudaGetDeviceCount( &nGpus ) );

    for ( int i = 0; i < nGpus; ++i )
    {
        CUDA_CHECK( cudaGetDeviceProperties( &prop, i ) );
        printf( "   --- General Information for device %d ---\n", i );
        printf( "Name:  %s\n", prop.name );
        printf( "Compute capability:  %d.%d\n", prop.major, prop.minor );
        printf( "Clock rate:  %d\n", prop.clockRate );
        printf( "Device copy overlap: %s\n", prop.deviceOverlap ? "Enabled" : "Disabled" );
        printf( "Kernel execution timeout : %s\n", prop.kernelExecTimeoutEnabled ? "Enabled" : "Disabled" );
        printf( "   --- Memory Information for device %d ---\n", i );
        printf( "Total global mem:  %ld\n", prop.totalGlobalMem );
        printf( "Total constant Mem:  %ld\n", prop.totalConstMem );
        printf( "Max mem pitch:  %ld\n", prop.memPitch );
        printf( "Texture Alignment:  %ld\n", prop.textureAlignment );

        printf( "   --- MP Information for device %d ---\n", i );
        printf( "Multiprocessor count:  %d\n", prop.multiProcessorCount );
        printf( "Shared mem per mp:  %ld\n", prop.sharedMemPerBlock );
        printf( "Registers per mp:  %d\n", prop.regsPerBlock );
        printf( "Threads in warp:  %d\n", prop.warpSize );
        printf( "Max threads per block:  %d\n", prop.maxThreadsPerBlock );
        printf( "Max thread dimensions:  (%d, %d, %d)\n",
                prop.maxThreadsDim[0],
                prop.maxThreadsDim[1],
                prop.maxThreadsDim[2] );
        printf( "Max grid dimensions:  (%d, %d, %d)\n",
                prop.maxGridSize[0],
                prop.maxGridSize[1],
                prop.maxGridSize[2] );
        printf( "\n" );
    }

    if ( iGpuToUse >= nGpus )
    {
        std::cout << "GPU with ID " << iGpuToUse << " not present. Only " << nGpus << " GPUs are available. Exiting..." << std::endl;
        throw std::runtime_error( "Can not find GPU or GPU not present. Exiting..." );
    }

    /* choose GPU to use */
    CUDA_CHECK( cudaSetDevice( iGpuToUse ));


    /**** create the BondTable and copy to constant memory ****/
    std::cout << "copy BondTable: " << std::endl;
    bool * tmpForbiddenBonds = (bool *) malloc(sizeof(bool)*512);
    uint nAllowedBonds = 0;
    for(int i = 0; i < 512; i++)
    {
        tmpForbiddenBonds[i] = false;
        tmpForbiddenBonds[i] = mForbiddenBonds[i];
        std::cout << "bond: " << i << "  " << tmpForbiddenBonds[i] << "  " << mForbiddenBonds[i] << std::endl;
        if ( ! tmpForbiddenBonds[i] )
            nAllowedBonds++;
    }
    std::cout << "used bond in simulation: " << nAllowedBonds << " / 108 " << std::endl;
    if ( nAllowedBonds != 108 )
        throw std::runtime_error( "wrong bond-set! Expected 108 allowed bonds. Exiting...\n" );
    CUDA_CHECK( cudaMemcpyToSymbol( dpForbiddenBonds, tmpForbiddenBonds, sizeof(bool)*512 ) );
    free(tmpForbiddenBonds);

    //creating the displacement arrays
    std::cout << "copy DXYZTable: " << std::endl;

    intCUDA tmp_DXTable[6] = { -1,1,  0,0,  0,0 };
    intCUDA tmp_DYTable[6] = {  0,0, -1,1,  0,0 };
    intCUDA tmp_DZTable[6] = {  0,0,  0,0, -1,1 };

    CUDA_CHECK(cudaMemcpyToSymbol(DXTable_d, tmp_DXTable, sizeof(intCUDA)*6));
    CUDA_CHECK(cudaMemcpyToSymbol(DYTable_d, tmp_DYTable, sizeof(intCUDA)*6));
    CUDA_CHECK(cudaMemcpyToSymbol(DZTable_d, tmp_DZTable, sizeof(intCUDA)*6));

    /***************************creating look-up for species*****************************************/

    /* count monomers per species before allocating per species arrays */
    uint32_t * pMonomerSpecies = (uint32_t *) malloc( nAllMonomers * sizeof(uint32_t) );
    nMonomersSpeciesA = 0;
    nMonomersSpeciesB = 0;
    for ( uint32_t i = 0; i < nAllMonomers; ++i )
    {
        // monomer is odd or even / A or B
        if ( mAttributeSystem[i] == 1 )
        {
            nMonomersSpeciesA++;
            pMonomerSpecies[i] = 1;
        }
        else if ( mAttributeSystem[i] == 2 )
        {
            nMonomersSpeciesB++;
            pMonomerSpecies[i] = 2;
        }
        else
            throw std::runtime_error( "wrong attributes!!! Exiting... \n" );
    }
    std::cout << "nMonomersSpezies_A: " << nMonomersSpeciesA << std::endl;
    std::cout << "nMonomersSpezies_B: " << nMonomersSpeciesB << std::endl;
    if ( nMonomersSpeciesA + nMonomersSpeciesB != nAllMonomers )
        throw std::runtime_error( "Nr Of MonomerSpezies does not add up! Exiting... \n");

    MonomersSpeziesIdx_A_host = (uint32_t *) malloc( nMonomersSpeciesA * sizeof(uint32_t) );
    MonomersSpeziesIdx_B_host = (uint32_t *) malloc( nMonomersSpeciesB * sizeof(uint32_t) );

    /* sort monomers (their indices) into corresponding species array  */
    uint32_t nMonomersWrittenA = 0;
    uint32_t nMonomersWrittenB = 0;
    for ( uint32_t i = 0; i < nAllMonomers; ++i )
    {
        if ( pMonomerSpecies[i] == 1 )
            MonomersSpeziesIdx_A_host[ nMonomersWrittenA++ ] = i;
        else if ( pMonomerSpecies[i] == 2 )
            MonomersSpeziesIdx_B_host[ nMonomersWrittenB++ ] = i;
    }
    if ( nMonomersSpeciesA != nMonomersWrittenA )
        throw std::runtime_error( "Number of monomers copeid for species A does not add up! Exiting... \n" );
    if ( nMonomersSpeciesB != nMonomersWrittenB )
        throw std::runtime_error( "Number of monomers copeid for species B does not add up! Exiting... \n" );

    /* move species tables to GPU */
    CUDA_CHECK( cudaMalloc((void **) &MonomersSpeziesIdx_A_device, (nMonomersSpeciesA)*sizeof(uint32_t)) );
    CUDA_CHECK( cudaMalloc((void **) &MonomersSpeziesIdx_B_device, (nMonomersSpeciesB)*sizeof(uint32_t)) );
    CUDA_CHECK( cudaMemcpy( MonomersSpeziesIdx_A_device, MonomersSpeziesIdx_A_host, (nMonomersSpeciesA)*sizeof(uint32_t), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy( MonomersSpeziesIdx_B_device, MonomersSpeziesIdx_B_host, (nMonomersSpeciesB)*sizeof(uint32_t), cudaMemcpyHostToDevice) );

    //make constant:
    CUDA_CHECK(cudaMemcpyToSymbol(nMonomersSpeciesA_d, &nMonomersSpeciesA, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(nMonomersSpeciesB_d, &nMonomersSpeciesB, sizeof(uint32_t)));

    /************************end: creating look-up for species*****************************************/

    /* calculate kernel config */
    numblocksSpecies_A = (nMonomersSpeciesA-1)/NUMTHREADS+1;
    std::cout << "calculate numBlocks Spezies A using" << (numblocksSpecies_A*NUMTHREADS) << "  needed: " << (nMonomersSpeciesA) <<  std::endl;
    numblocksSpecies_B = (nMonomersSpeciesB-1)/NUMTHREADS+1;
    std::cout << "calcluate numBlocks Spezies B using" << (numblocksSpecies_B*NUMTHREADS) << "  needed: " << (nMonomersSpeciesB) <<  std::endl;

    /****************************copy monomer informations ********************************************/


    mPolymerSystem_host =(intCUDA *) malloc((4*nAllMonomers+1)*sizeof(intCUDA));

    std::cout << "try to allocate : " << ((4*nAllMonomers+1)*sizeof(intCUDA)) << " bytes = " << ((4*nAllMonomers+1)*sizeof(intCUDA)/(1024.0)) << " kB = " << ((4*nAllMonomers+1)*sizeof(intCUDA)/(1024.0*1024.0)) << " MB coordinates on GPU " << std::endl;

    CUDA_CHECK(cudaMalloc((void **) &mPolymerSystem_device, (4*nAllMonomers+1)*sizeof(intCUDA)));


    for (uint32_t i=0; i<nAllMonomers; i++)
    {
        mPolymerSystem_host[4*i]=(intCUDA) mPolymerSystem[3*i];
        mPolymerSystem_host[4*i+1]=(intCUDA) mPolymerSystem[3*i+1];
        mPolymerSystem_host[4*i+2]=(intCUDA) mPolymerSystem[3*i+2];
        mPolymerSystem_host[4*i+3]=0;
    }

    // prepare and copy the connectivity matrix to GPU
    // the index on GPU starts at 0 and is one less than loaded

    int sizeMonoInfo = nAllMonomers * sizeof(MonoInfo);

    std::cout << "size of strut MonoInfo: " << sizeof(MonoInfo) << " bytes = " << (sizeof(MonoInfo)/(1024.0)) <<  "kB for one monomer connectivity " << std::endl;

    std::cout << "try to allocate : " << (sizeMonoInfo) << " bytes = " << (sizeMonoInfo/(1024.0)) <<  "kB = " << (sizeMonoInfo/(1024.0*1024.0)) <<  "MB for connectivity matrix on GPU " << std::endl;


    MonoInfo_host=(MonoInfo*) calloc(nAllMonomers,sizeof(MonoInfo));
    CUDA_CHECK(  cudaMalloc((void **) &MonoInfo_device, sizeMonoInfo));   // Allocate array of structure on device


    for ( uint32_t i = 0; i < nAllMonomers; ++i )
    {
        //MonoInfo_host[i].size = monosNNidx[i]->size;
        if((monosNNidx[i]->size) > 7)
        {
            std::cout << "this GPU-model allows max 7 next neighbors but size is " << (monosNNidx[i]->size) << ". Exiting..." << std::endl;
            throw std::runtime_error("Limit of connectivity on GPU reached!!! Exiting... \n");
        }

        mPolymerSystem_host[4*i+3] |= ((intCUDA)(monosNNidx[i]->size)) << 5;
        //cout << "mono:" << i << " vs " << (i) << endl;
        //cout << "numElements:" << MonoInfo_host[i].size << " vs " << monosNNidx[i]->size << endl;

        for(unsigned u=0; u < MAX_CONNECTIVITY; u++)
        {
            MonoInfo_host[i].bondsMonomerIdx[u] = monosNNidx[i]->bondsMonomerIdx[u];

            //cout << "bond["<< u << "]: " << MonoInfo_host[i].bondsMonomerIdx[u] << " vs " << monosNNidx[i]->bondsMonomerIdx[u] << endl;
        }
    }

    // copy to connectivity to device
    CUDA_CHECK( cudaMemcpy(MonoInfo_device, MonoInfo_host, sizeMonoInfo, cudaMemcpyHostToDevice));

    /****************************end: copy monomer informations ****************************************/

    checkSystem();

    /****************************creating lattice******************************************************/

    uint32_t LATTICE_X = Box_X;
    uint32_t LATTICE_Y = Box_Y;
    uint32_t LATTICE_Z = Box_Z;

    uint32_t LATTICE_XM1 = Box_XM1;
    uint32_t LATTICE_YM1 = Box_YM1;
    uint32_t LATTICE_ZM1 = Box_ZM1;

    uint32_t LATTICE_XPRO = Box_XPRO;
    uint32_t LATTICE_PROXY = Box_PROXY;

    CUDA_CHECK(cudaMemcpyToSymbol(LATTICE_X_d, &LATTICE_X, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(LATTICE_Y_d, &LATTICE_Y, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(LATTICE_Z_d, &LATTICE_Z, sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpyToSymbol(LATTICE_XM1_d, &LATTICE_XM1, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(LATTICE_YM1_d, &LATTICE_YM1, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(LATTICE_ZM1_d, &LATTICE_ZM1, sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpyToSymbol(LATTICE_XPRO_d, &LATTICE_XPRO, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(LATTICE_PROXY_d, &LATTICE_PROXY, sizeof(uint32_t)));


    mLatticeOut_host = (uint8_t *) malloc( LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t));

    mLatticeTmp_host = (uint8_t *) malloc( LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t));

    std::cout << "try to allocate : " << (LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t)) << " bytes = " << (LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t)/(1024.0*1024.0)) << " MB lattice on GPU " << std::endl;


    CUDA_CHECK(cudaMalloc((void **) &mLatticeOut_device, LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc((void **) &mLatticeTmp_device, LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t)));


    //copy information from Host to GPU
    for(int i = 0; i < LATTICE_X*LATTICE_Y*LATTICE_Z; i++)
    {
        mLatticeOut_host[i]=0;
        mLatticeTmp_host[i]=0;

    }

    //fill the tmpmLattice - should be zero everywhere
    CUDA_CHECK(cudaMemcpy(mLatticeTmp_device, mLatticeTmp_host, LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t), cudaMemcpyHostToDevice));
    //start z-curve
    /*
    for (int t = 0; t < nAllMonomers; t++)
    {
        uint32_t xk = (mPolymerSystem[3*t  ]&LATTICE_XM1);
        uint32_t yk = (mPolymerSystem[3*t+1]&LATTICE_YM1);
        uint32_t zk = (mPolymerSystem[3*t+2]&LATTICE_ZM1);

        uint32_t inter3 = interleave3(xk/2,yk/2,zk/2);

        mLatticeOut_host[((mPolymerSystem_host[4*t+3] & 1) << 23) +inter3] = 1;
    }
    */
    //end- z-curve

    for (int t = 0; t < nAllMonomers; t++)
    {
        uint32_t xk = (mPolymerSystem[3*t  ]&LATTICE_XM1);
        uint32_t yk = (mPolymerSystem[3*t+1]&LATTICE_YM1);
        uint32_t zk = (mPolymerSystem[3*t+2]&LATTICE_ZM1);

        //uint32_t inter3 = interleave3(xk/2,yk/2,zk/2);

        mLatticeOut_host[xk + (yk << LATTICE_XPRO) + (zk << LATTICE_PROXY)] = 1;
    }

    std::cout << "checking the  mLatticeOut_host: " << std::endl;
    CUDA_CHECK(cudaMemcpy(mLatticeOut_device, mLatticeOut_host, LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t), cudaMemcpyHostToDevice));


    //fetch from device and check again
    for(int i = 0; i < LATTICE_X*LATTICE_Y*LATTICE_Z; i++)
            mLattice[i]=0;

    std::cout << "copy back mLatticeOut_host: " << std::endl;
    CUDA_CHECK(cudaMemcpy(mLatticeOut_host, mLatticeOut_device, LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t), cudaMemcpyDeviceToHost));

    for(int i = 0; i < LATTICE_X*LATTICE_Y*LATTICE_Z; i++)
        mLattice[i]=mLatticeOut_host[i];

    std::cout << "copy back mLatticeTmp_host: " << std::endl;
    CUDA_CHECK(cudaMemcpy(mLatticeTmp_host, mLatticeTmp_device, LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t), cudaMemcpyDeviceToHost));

    int dummyTmpCounter=0;
    for (int x=0;x<LATTICE_X;x++)
    for (int y=0;y<LATTICE_Y;y++)
    for (int z=0;z<LATTICE_Z;z++)
        dummyTmpCounter += ( mLatticeTmp_host[x + (y << LATTICE_XPRO) + ( z << LATTICE_PROXY)] == 0 ) ? 0 : 1;

    std::cout << "occupied latticeTmp sites: " << dummyTmpCounter << " of " << (0) << std::endl;

    if(dummyTmpCounter != 0)
        throw std::runtime_error("mLattice occupation is wrong!!! Exiting... \n");

    //start -z-order
    /*
    cout << "recalculate mLattice: " << endl;
    //fetch from device and check again
    for(int i = 0; i < LATTICE_X*LATTICE_Y*LATTICE_Z; i++)
    {
        if(mLatticeOut_host[i]==1)
        {
            uint32_t dummyhost = i;
            uint32_t onX = (dummyhost / (1 <<23)); //0 on O, 1 on X
            uint32_t zl = 2*( deinterleave3_Z((dummyhost % (1 <<23)))) + onX;
            uint32_t yl = 2*( deinterleave3_Y((dummyhost % (1 <<23)))) + onX;
            uint32_t xl = 2*( deinterleave3_X((dummyhost % (1 <<23)))) + onX;


            //cout << "X: " << xl << "\tY: " << yl << "\tZ: " << zl<< endl;
            mLattice[xl + (yl << LATTICE_XPRO) + (zl << LATTICE_PROXY)] = 1;

        }

    }
    */
    //end -z-order


    /*************************end: creating lattice****************************************************/


    /*************************copy monomer positions***************************************************/
    CUDA_CHECK(cudaMemcpy(mPolymerSystem_device, mPolymerSystem_host, (4*nAllMonomers+1)*sizeof(intCUDA), cudaMemcpyHostToDevice));
    /*************************end: copy monomer positions**********************************************/

    /*************************bind textures on GPU ****************************************************/
    std::cout << "bind textures "  << std::endl;
    //bind texture reference with linear memory
    cudaBindTexture(0,texmLatticeRefOut,mLatticeOut_device,LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t));
    cudaBindTexture(0,texmLatticeTmpRef,mLatticeTmp_device,LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t));
    cudaBindTexture(0,texPolymerAndMonomerIsEvenAndOnXRef,mPolymerSystem_device,(4*nAllMonomers+1)*sizeof(intCUDA));
    cudaBindTexture(0,texMonomersSpezies_A_ThreadIdx,MonomersSpeziesIdx_A_device,(nMonomersSpeciesA)*sizeof(uint32_t));
    cudaBindTexture(0,texMonomersSpezies_B_ThreadIdx,MonomersSpeziesIdx_B_device,(nMonomersSpeciesB)*sizeof(uint32_t));

    /* new with texture object... they said it would be easier -.- */
    cudaResourceDesc resDescA;
    memset( &resDescA, 0, sizeof( resDescA ) );
    resDescA.resType                = cudaResourceTypeLinear;
    resDescA.res.linear.desc.f      = cudaChannelFormatKindUnsigned;
    resDescA.res.linear.desc.x      = 32; // bits per channel
    cudaResourceDesc resDescB = resDescA;
    resDescA.res.linear.devPtr      = MonomersSpeziesIdx_A_device;
    resDescA.res.linear.sizeInBytes = nMonomersSpeciesA * sizeof( uint32_t );
    resDescB.res.linear.devPtr      = MonomersSpeziesIdx_B_device;
    resDescB.res.linear.sizeInBytes = nMonomersSpeciesB * sizeof( uint32_t );

    cudaTextureDesc texDescA;
    memset( &texDescA, 0, sizeof( texDescA ) );
    texDescA.readMode = cudaReadModeElementType;
    cudaTextureDesc texDescB = texDescA;

    cudaCreateTextureObject( &texSpeciesIndicesA, &resDescA, &texDescA, NULL );
    cudaCreateTextureObject( &texSpeciesIndicesB, &resDescB, &texDescB, NULL );


    /*************************end: bind textures on GPU ************************************************/

    /*************************last check of system GPU *************************************************/

    CUDA_CHECK(cudaMemcpy(mPolymerSystem_host, mPolymerSystem_device, (4*nAllMonomers+1)*sizeof(intCUDA), cudaMemcpyDeviceToHost));

    for( uint32_t i = 0; i < nAllMonomers; ++i )
    {
        mPolymerSystem[3*i+0] = (int32_t) mPolymerSystem_host[4*i+0];
        mPolymerSystem[3*i+1] = (int32_t) mPolymerSystem_host[4*i+1];
        mPolymerSystem[3*i+2] = (int32_t) mPolymerSystem_host[4*i+2];
    }

    /* why two times ??? */
    std::cout << "check system before simulation: " << std::endl;
    checkSystem();

    std::cout << "check system before simulation: " << std::endl;
    checkSystem();

    /*************************end: last check of system GPU *********************************************/
}

/**
 * !!! Problems:
 *  Note that this simply bitmasks negative values, e.g. x=-4 = 0xfffffffc
 *  Note that 0xfc = 1111 1100b and &7 -> 100. Vice-versa 4 = 100b ...
 *   => this clashes !!! As a simple runtime test shows, both are indeed used!
 */
int UpdaterGPUScBFM_AB_Type::IndexBondArray( int const x, int const y, int const z )
{
#ifndef NDEBUG2
    if ( x == -4 || x == 4 )
    {
        /* Found negative x=-4 = fffffffc */
        std::cout << "[" << __FILENAME__ << "::IndexBondArray] +-4 x="
                  << x << " = " << std::hex << x << std::dec << std::endl;
    }
#endif
    /* 7 == 0b111, i.e. truncate the lowest 3 bits */
    return   ( x & 7 ) +
           ( ( y & 7 ) << 3 ) +
           ( ( z & 7 ) << 6 );
}

void UpdaterGPUScBFM_AB_Type::setNrOfAllMonomers( uint32_t rnAllMonomers )
{
    nAllMonomers = rnAllMonomers;
    std::cout << "[" << __FILENAME__ << "::setNrOfAllMonomers" << "] used monomers in simulation: " << nAllMonomers << std::endl;

    mAttributeSystem = new int32_t[nAllMonomers];
    mPolymerSystem   = new int32_t[nAllMonomers*3+1];    /* why +1 ??? */

    //idx is reduced by one compared to the file
    monosNNidx = new MonoNNIndex*[nAllMonomers];
    for ( uint32_t a = 0; a < nAllMonomers; ++a )
    {
        monosNNidx[a] = new MonoNNIndex();
        monosNNidx[a]->size=0;
        for ( unsigned o = 0; o < MAX_CONNECTIVITY; ++o )
            monosNNidx[a]->bondsMonomerIdx[o]=0;
    }
}

void UpdaterGPUScBFM_AB_Type::setPeriodicity(bool isPeriodicX, bool isPeriodicY, bool isPeriodicZ)
{
    //check if we are using periodic boundary condition and the simulations are do so
#ifdef NONPERIODICITY
    if((isPeriodicX == true) || (isPeriodicY == true) || (isPeriodicZ == true) )
    {
        std::stringstream errormessage;
        errormessage<<"Simulation is intended to use NON-PERIODIC BOUNDARY conditions.\n";
        errormessage<<"But in BFM-File the PERIODICITY is set to:\n";
        errormessage<<"In X:"<<isPeriodicX<<"\n";
        errormessage<<"In Y:"<<isPeriodicY<<"\n";
        errormessage<<"In Z:"<<isPeriodicZ<<"\n";
        errormessage<<"Logical Error! Exiting...\n";
        throw std::runtime_error(errormessage.str());
    }
#else
    if((isPeriodicX == false) || (isPeriodicY == false) || (isPeriodicZ == false) )
    {
        std::stringstream errormessage;
        errormessage<<"Simulation is intended to use PERIODIC BOUNDARY conditions.\n";
        errormessage<<"But in BFM-File the PERIODICITY is set to:\n";
        errormessage<<"In X:"<<isPeriodicX<<"\n";
        errormessage<<"In Y:"<<isPeriodicY<<"\n";
        errormessage<<"In Z:"<<isPeriodicZ<<"\n";
        errormessage<<"Logical Error! Exiting...\n";
        throw std::runtime_error(errormessage.str());
    }
#endif

}

void UpdaterGPUScBFM_AB_Type::setNetworkIngredients(uint32_t numPEG, uint32_t numPEGArm, uint32_t numCL)
{
    nStars = numPEG; //number of Stars
    nMonomersPerStarArm = numPEGArm; //number OfMonomersPerStarArm
    nCrosslinker = numCL; //number of Crosslinker

    std::cout << "NumPEG on GPU: " << nStars << std::endl;
    std::cout << "NumPEGArmlength on GPU: " << nMonomersPerStarArm << std::endl;
    std::cout << "NumCrosslinker on GPU: " << nCrosslinker << std::endl;

    //if (nMonomersPerStarArm != 29)
        //throw std::runtime_error("nMonomersPerStarArm should be 29!!! Exiting...\n");

    //if ((nMonomersPerStarArm%2) != 1)
        //    throw std::runtime_error("nMonomersPerStarArm should be an odd number!!! Exiting...\n");

}

void UpdaterGPUScBFM_AB_Type::setConnectivity(uint32_t monoidx1, uint32_t monoidx2)
{
    monosNNidx[monoidx1]->bondsMonomerIdx[monosNNidx[monoidx1]->size] = monoidx2;
    //monosNNidx[monoidx2]->bondsMonomerIdx[monosNNidx[monoidx2]->size] = monoidx1;

    monosNNidx[monoidx1]->size++;
    //monosNNidx[monoidx2]->size++;

    //if((monosNNidx[monoidx1]->size > MAX_CONNECTIVITY) || (monosNNidx[monoidx2]->size > MAX_CONNECTIVITY))
    if ( monosNNidx[monoidx1]->size > MAX_CONNECTIVITY )
        throw std::runtime_error("MAX_CONNECTIVITY  exceeded!!! Exiting...\n");
}

void UpdaterGPUScBFM_AB_Type::setLatticeSize
(
    uint32_t const boxX,
    uint32_t const boxY,
    uint32_t const boxZ
)
{
    Box_X = boxX;
    Box_Y = boxY;
    Box_Z = boxZ;

    Box_XM1 = boxX-1;
    Box_YM1 = boxY-1;
    Box_ZM1 = boxZ-1;

    // determine the shift values for first multiplication
    uint32_t resultshift = -1;
    uint32_t dummy = boxX;
    while ( dummy != 0 )
    {
        dummy >>= 1;
        resultshift++;
    }
    Box_XPRO=resultshift;

    // determine the shift values for first multiplication
    resultshift = -1;
    dummy = boxX*boxY;
    while ( dummy != 0 )
    {
        dummy >>= 1;
        resultshift++;
    }
    Box_PROXY=resultshift;

    std::cout << "use bit shift for boxX: (1 << "<< Box_XPRO << " ) = " << (1 << Box_XPRO) << " = " << (boxX) << std::endl;
    std::cout << "use bit shift for boxX*boxY: (1 << "<< Box_PROXY << " ) = " << (1 << Box_PROXY) << " = " << (boxX*boxY) << std::endl;

    // check if shift is correct
    if ( (boxX != (1 << Box_XPRO)) || ((boxX*boxY) != (1 << Box_PROXY)) )
        throw  std::runtime_error( "Could not determine value for bit shift. Sure your box size is a power of 2? Exiting...\n" );

    //init lattice
    mLattice = new uint8_t[Box_X*Box_Y*Box_Z];

    for(int i = 0; i < Box_X*Box_Y*Box_Z; i++)
        mLattice[i]=0;
}

void UpdaterGPUScBFM_AB_Type::populateLattice()
{
    //if(!GPUScBFM.StartSimulationGPU())
    for ( size_t i = 0; i < nAllMonomers; ++i )
    {
        mLattice[  ( mPolymerSystem[3*i+0] & Box_XM1) +
                 ( ( mPolymerSystem[3*i+1] & Box_YM1 ) << Box_XPRO ) +
                 ( ( mPolymerSystem[3*i+2] & Box_ZM1 ) << Box_PROXY) ] = 1;
    }
}

void UpdaterGPUScBFM_AB_Type::checkSystem()
{
    std::cout << "checkSystem" << std::endl;

    int countermLatticeStart = 0;

        //if(!GPUScBFM.StartSimulationGPU())
    for(int i = 0; i < Box_X*Box_Y*Box_Z; i++)
        mLattice[i]=0;

    for (int idxMono=0; idxMono < (nAllMonomers); idxMono++)
    {
        int32_t xpos = mPolymerSystem[3*idxMono    ];
        int32_t ypos = mPolymerSystem[3*idxMono+1  ];
        int32_t zpos = mPolymerSystem[3*idxMono+2  ];

        mLattice[((0 + xpos) & Box_XM1) + (((0 + ypos) & Box_YM1)<< Box_XPRO) + (((0 + zpos) & Box_ZM1)<< Box_PROXY)]=1;
        mLattice[((1 + xpos) & Box_XM1) + (((0 + ypos) & Box_YM1)<< Box_XPRO) + (((0 + zpos) & Box_ZM1)<< Box_PROXY)]=1;
        mLattice[((0 + xpos) & Box_XM1) + (((1 + ypos) & Box_YM1)<< Box_XPRO) + (((0 + zpos) & Box_ZM1)<< Box_PROXY)]=1;
        mLattice[((1 + xpos) & Box_XM1) + (((1 + ypos) & Box_YM1)<< Box_XPRO) + (((0 + zpos) & Box_ZM1)<< Box_PROXY)]=1;

        mLattice[((0 + xpos) & Box_XM1) + (((0 + ypos) & Box_YM1)<< Box_XPRO) + (((1 + zpos) & Box_ZM1)<< Box_PROXY)]=1;
        mLattice[((1 + xpos) & Box_XM1) + (((0 + ypos) & Box_YM1)<< Box_XPRO) + (((1 + zpos) & Box_ZM1)<< Box_PROXY)]=1;
        mLattice[((0 + xpos) & Box_XM1) + (((1 + ypos) & Box_YM1)<< Box_XPRO) + (((1 + zpos) & Box_ZM1)<< Box_PROXY)]=1;
        mLattice[((1 + xpos) & Box_XM1) + (((1 + ypos) & Box_YM1)<< Box_XPRO) + (((1 + zpos) & Box_ZM1)<< Box_PROXY)]=1;

    }

    for (int x=0;x<Box_X;x++)
    for (int y=0;y<Box_Y;y++)
    for (int z=0;z<Box_Z;z++)
    {
       countermLatticeStart += (mLattice[x + (y << Box_XPRO) + (z << Box_PROXY)]==0)? 0 : 1;
       //if (mLattice[x + (y << LATTICE_XPRO) + (z << LATTICE_PROXY)] != 0)
       //cout << x << " " << y << " " << z << "\t" <<  mLattice[x + (y << LATTICE_XPRO) + (z << LATTICE_PROXY)]<< endl;

    }
    //countermLatticeStart *=8;

    std::cout << "occupied lattice sites: " << countermLatticeStart << " of " << (nAllMonomers*8) << std::endl;

    if(countermLatticeStart != (nAllMonomers*8))
        throw std::runtime_error("mLattice occupation is wrong!!! Exiting... \n");


    std::cout << "check bonds" << std::endl;

    for (int idxMono=0; idxMono < (nAllMonomers); idxMono++)
    for(unsigned idxNN=0; idxNN < monosNNidx[idxMono]->size; idxNN++)
    {
         int32_t bond_x = (mPolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]  ]-mPolymerSystem[3*idxMono  ]);
         int32_t bond_y = (mPolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+1]-mPolymerSystem[3*idxMono+1]);
         int32_t bond_z = (mPolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+2]-mPolymerSystem[3*idxMono+2]);

         if((bond_x > 3) || (bond_x < -3))
         {
             std::cout << "Invalid XBond..."<< std::endl;
             std::cout << bond_x<< " " << bond_y<< " " << bond_z<< "  between mono: " <<(idxMono+1)<< " (pos "<< mPolymerSystem[3*idxMono  ] <<","<<mPolymerSystem[3*idxMono+1]<<","<<mPolymerSystem[3*idxMono+2] <<") and " << (monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+1) << " (pos "<< mPolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]  ] <<","<<mPolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+1]<<","+mPolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+2] <<")"<<std::endl;
             throw std::runtime_error("Invalid XBond!!! Exiting...\n");
         }

         if((bond_y > 3) || (bond_y < -3))
         {
             std::cout << "Invalid YBond..."<< std::endl;
             std::cout << bond_x<< " " << bond_y<< " " << bond_z<< "  between mono: " <<(idxMono+1)<< " (pos "<< mPolymerSystem[3*idxMono  ] <<","<<mPolymerSystem[3*idxMono+1]<<","<<mPolymerSystem[3*idxMono+2] <<") and " << (monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+1) << " (pos "<< mPolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]  ] <<","<<mPolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+1]<<","+mPolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+2] <<")"<<std::endl;
             throw std::runtime_error("Invalid YBond!!! Exiting...\n");

         }

         if((bond_z > 3) || (bond_z < -3))
         {
             std::cout << "Invalid ZBond..."<< std::endl;
             std::cout << bond_x<< " " << bond_y<< " " << bond_z<< "  between mono: " <<(idxMono+1)<< " (pos "<< mPolymerSystem[3*idxMono  ] <<","<<mPolymerSystem[3*idxMono+1]<<","<<mPolymerSystem[3*idxMono+2] <<") and " << (monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+1) << " (pos "<< mPolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]  ] <<","<<mPolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+1]<<","+mPolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+2] <<")"<<std::endl;
             throw std::runtime_error("Invalid ZBond!!! Exiting...\n");
         }


         //false--erlaubt; true-forbidden
        if( mForbiddenBonds[IndexBondArray(bond_x, bond_y, bond_z)] )
        {
            std::cout << "something wrong with bonds between monomer: " << monosNNidx[idxMono]->bondsMonomerIdx[idxNN]  << " and " << idxMono << std::endl;
            std::cout << (monosNNidx[idxMono]->bondsMonomerIdx[idxNN]) << "\t x: " << (mPolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]])   << "\t y:" << mPolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+1]<< "\t z:" << mPolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+2]<< std::endl;
            std::cout << idxMono << "\t x:" << mPolymerSystem[3*idxMono  ] << "\t y:" << mPolymerSystem[3*idxMono+1]<< "\t z:" << mPolymerSystem[3*idxMono  ]<< std::endl;

            throw std::runtime_error("Bond is NOT allowed!!! Exiting...\n");
        }

    }
}

void UpdaterGPUScBFM_AB_Type::runSimulationOnGPU( int32_t nrMCS )
{
    std::clock_t const t0 = std::clock();

    //run simulation
    for ( int32_t timeS = 1; timeS <= nrMCS; ++timeS )
    {
        /******* OneMCS ******/
        for(uint32_t cou = 0; cou < 2; cou++)
        {
            switch(randomNumbers.r250_rand32()%2)
            {
                case 0:  // run Spezies_A monomers
                        kernelSimulationScBFMCheckSpezies<<<numblocksSpecies_A,NUMTHREADS>>>(mPolymerSystem_device, mLatticeTmp_device, MonoInfo_device, texSpeciesIndicesA, nMonomersSpeciesA, randomNumbers.r250_rand32());
                        runSimulationScBFMPerformSpeziesA_gpu<<<numblocksSpecies_A,NUMTHREADS>>>(mPolymerSystem_device, mLatticeOut_device);
                        runSimulationScBFMZeroArraySpeziesA_gpu<<<numblocksSpecies_A,NUMTHREADS>>>(mPolymerSystem_device, mLatticeTmp_device);
                        break;

                case 1: // run Spezies_B monomers
                        kernelSimulationScBFMCheckSpezies<<<numblocksSpecies_B,NUMTHREADS>>>(mPolymerSystem_device, mLatticeTmp_device, MonoInfo_device, texSpeciesIndicesB, nMonomersSpeciesB, randomNumbers.r250_rand32());
                        runSimulationScBFMPerformSpeziesB_gpu<<<numblocksSpecies_B,NUMTHREADS>>>(mPolymerSystem_device, mLatticeOut_device);
                        runSimulationScBFMZeroArraySpeziesB_gpu<<<numblocksSpecies_B,NUMTHREADS>>>(mPolymerSystem_device, mLatticeTmp_device);
                        break;

                default: break;
            }
        }

        /*
        if ((timeS%saveTime==0))
        {
            //copy information from GPU to Host

            //check the tmpmLattice
            cout << "copy back mLatticeTmp_host: " << endl;
            CUDA_CHECK(cudaMemcpy(mLatticeTmp_host, mLatticeTmp_device, LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t), cudaMemcpyDeviceToHost));

            int dummyTmpCounter=0;
            for (int x=0;x<LATTICE_X;x++)
                for (int y=0;y<LATTICE_Y;y++)
                    for (int z=0;z<LATTICE_Z;z++)
                             {
                                dummyTmpCounter += (mLatticeTmp_host[x + (y << LATTICE_XPRO) + (z << LATTICE_PROXY)]==0)? 0 : 1;
                             }
            cout << "occupied latticeTmp sites: " << dummyTmpCounter << " of " << (0) << endl;



            CUDA_CHECK(cudaMemcpy(mLatticeOut_host, mLatticeOut_device, LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t), cudaMemcpyDeviceToHost));

            for(int i = 0; i < LATTICE_X*LATTICE_Y*LATTICE_Z; i++)
                    mLattice[i]=0;

            for(int i = 0; i < LATTICE_X*LATTICE_Y*LATTICE_Z; i++)
                mLattice[i]=mLatticeOut_host[i];

            if(dummyTmpCounter != 0)
                    exit(-1);

            //start -z-order
            //
            //cout << "save -- recalculate mLattice: " << endl;
            //fetch from device and check again
            //    for(int i = 0; i < LATTICE_X*LATTICE_Y*LATTICE_Z; i++)
            //    {
            //        if(mLatticeOut_host[i]==1)
            //        {
            //            uint32_t dummyhost = i;
            //            uint32_t onX = (dummyhost / (1 <<23)); //0 on O, 1 on X
            //            uint32_t zl = 2*( deinterleave3_Z((dummyhost % (1 <<23)))) + onX;
            //            uint32_t yl = 2*( deinterleave3_Y((dummyhost % (1 <<23)))) + onX;
            //            uint32_t xl = 2*( deinterleave3_X((dummyhost % (1 <<23)))) + onX;


                        //cout << "X: " << xl << "\tY: " << yl << "\tZ: " << zl<< endl;
            //            mLattice[xl + (yl << LATTICE_XPRO) + (zl << LATTICE_PROXY)] = 1;
                        //
            //        }
                    //
            //    }
                //end -z-order
            //


            CUDA_CHECK(cudaMemcpy(mPolymerSystem_host, mPolymerSystem_device, (4*nAllMonomers+1)*sizeof(intCUDA), cudaMemcpyDeviceToHost));

            for (uint32_t i=0; i<nAllMonomers; i++)
            {
                mPolymerSystem[3*i]=(int32_t) mPolymerSystem_host[4*i];
                mPolymerSystem[3*i+1]=(int32_t) mPolymerSystem_host[4*i+1];
                mPolymerSystem[3*i+2]=(int32_t) mPolymerSystem_host[4*i+2];
                //cout << i << "  : " << mPolymerSystem_host[4*i+3] << endl;
            }

            checkSystem();


            SaveSystem(timeS);
            cout << "actual time: " << timeS << endl;

            difference = time(NULL) - startTimer;
            cout << "mcs = " << (timeS+MCSTime)  << "  speed [performed monomer try and move/s] = MCS*N/t: " << (1.0 * timeS * ((1.0 * nAllMonomers) / (1.0f * difference))) << "     runtime[s]:" << (1.0f * difference) << endl;


        }
        */
    }

    //All MCS are done- copy back...


    //copy information from GPU to Host

    //check the tmpmLattice
    std::cout << "copy back mLatticeTmp_host: " << std::endl;
    CUDA_CHECK(cudaMemcpy(mLatticeTmp_host, mLatticeTmp_device, Box_X*Box_Y*Box_Z*sizeof(uint8_t), cudaMemcpyDeviceToHost));

    int dummyTmpCounter=0;
    for ( int x = 0; x < Box_X; ++x )
    for ( int y = 0; y < Box_Y; ++y )
    for ( int z = 0; z < Box_Z; ++z )
        dummyTmpCounter += (mLatticeTmp_host[x + (y << Box_XPRO) + (z << Box_PROXY)]==0)? 0 : 1;
    std::cout << "occupied latticeTmp sites: " << dummyTmpCounter << " of " << (0) << std::endl;


    CUDA_CHECK(cudaMemcpy(mLatticeOut_host, mLatticeOut_device, Box_X*Box_Y*Box_Z*sizeof(uint8_t), cudaMemcpyDeviceToHost));

    /* why set it to 0 if it will have values written to it right after that
     * anyway ??? */
    for ( int i = 0; i < Box_X * Box_Y * Box_Z; ++i )
        mLattice[i] = 0;

    for ( int i = 0; i < Box_X * Box_Y * Box_Z; ++i )
        mLattice[i] = mLatticeOut_host[i];

    if ( dummyTmpCounter != 0 )
        throw std::runtime_error("mLattice occupation is wrong!!! Exiting... \n");

    //start -z-order
    /*
    cout << "save -- recalculate mLattice: " << endl;
    //fetch from device and check again
        for(int i = 0; i < LATTICE_X*LATTICE_Y*LATTICE_Z; i++)
        {
            if(mLatticeOut_host[i]==1)
            {
                uint32_t dummyhost = i;
                uint32_t onX = (dummyhost / (1 <<23)); //0 on O, 1 on X
                uint32_t zl = 2*( deinterleave3_Z((dummyhost % (1 <<23)))) + onX;
                uint32_t yl = 2*( deinterleave3_Y((dummyhost % (1 <<23)))) + onX;
                uint32_t xl = 2*( deinterleave3_X((dummyhost % (1 <<23)))) + onX;


                //cout << "X: " << xl << "\tY: " << yl << "\tZ: " << zl<< endl;
                mLattice[xl + (yl << LATTICE_XPRO) + (zl << LATTICE_PROXY)] = 1;

            }

        }
        //end -z-order
    */


    CUDA_CHECK(cudaMemcpy(mPolymerSystem_host, mPolymerSystem_device, (4*nAllMonomers+1)*sizeof(intCUDA), cudaMemcpyDeviceToHost));

    for ( uint32_t i = 0; i < nAllMonomers; ++i )
    {
        mPolymerSystem[ 3*i+0 ] = (int32_t) mPolymerSystem_host[ 4*i+0 ];
        mPolymerSystem[ 3*i+1 ] = (int32_t) mPolymerSystem_host[ 4*i+1 ];
        mPolymerSystem[ 3*i+2 ] = (int32_t) mPolymerSystem_host[ 4*i+2 ];
    }

    checkSystem();

    std::clock_t const t1 = std::clock();
    double const dt = float(t1-t0) / CLOCKS_PER_SEC;
    std::cout
    << "run time (GPU): " << nrMCS << "\n"
    << "mcs = " << (nrMCS)  << "  speed [performed monomer try and move/s] = MCS*N/t: "
    << nrMCS * ( nAllMonomers / dt )  << "     runtime[s]:" << dt << std::endl;
}

void UpdaterGPUScBFM_AB_Type::cleanup()
{
    // copy information from GPU to Host
    CUDA_CHECK( cudaMemcpy(mLattice, mLatticeOut_device, Box_X*Box_Y*Box_Z*sizeof(uint8_t), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(mPolymerSystem_host, mPolymerSystem_device, (4*nAllMonomers+1)*sizeof(intCUDA), cudaMemcpyDeviceToHost) );

    for (uint32_t i=0; i<nAllMonomers; i++)
    {
        mPolymerSystem[3*i]=(int32_t) mPolymerSystem_host[4*i];
        mPolymerSystem[3*i+1]=(int32_t) mPolymerSystem_host[4*i+1];
        mPolymerSystem[3*i+2]=(int32_t) mPolymerSystem_host[4*i+2];
    }

    checkSystem();

    int sizeMonoInfo = nAllMonomers * sizeof(MonoInfo);
    // copy connectivity matrix back from device to host
    CUDA_CHECK( cudaMemcpy(MonoInfo_host, MonoInfo_device, sizeMonoInfo, cudaMemcpyDeviceToHost));

    for (uint32_t i=0; i<nAllMonomers; i++)
    {

        //if(MonoInfo_host[i].size != monosNNidx[i]->size)
        if(((mPolymerSystem_host[4*i+3]&224)>>5) != monosNNidx[i]->size)
        {
            std::cout << "connectivity error after simulation run" << std::endl;
            std::cout << "mono:" << i << " vs " << (i) << std::endl;
            //cout << "numElements:" << MonoInfo_host[i].size << " vs " << monosNNidx[i]->size << endl;
            std::cout << "numElements:" << ((mPolymerSystem_host[4*i+3]&224)>>5) << " vs " << monosNNidx[i]->size << std::endl;

            throw std::runtime_error("Connectivity is corrupted!!! Maybe your Simulation is wrong!!! Exiting...\n");
        }

        for(unsigned u=0; u < MAX_CONNECTIVITY; u++)
        {
            if(MonoInfo_host[i].bondsMonomerIdx[u] != monosNNidx[i]->bondsMonomerIdx[u])
            {
                std::cout << "connectivity error after simulation run" << std::endl;
                std::cout << "mono:" << i << " vs " << (i) << std::endl;

                std::cout << "bond["<< u << "]: " << MonoInfo_host[i].bondsMonomerIdx[u] << " vs " << monosNNidx[i]->bondsMonomerIdx[u] << std::endl;

                throw std::runtime_error("Connectivity is corrupted!!! Maybe your Simulation is wrong!!! Exiting...\n");
            }
        }
    }

    std::cout << "no errors in connectivity matrix after simulation run" << std::endl;


    checkSystem();

    //unbind texture reference to free resource
    cudaUnbindTexture( texmLatticeRefOut                   );
    cudaUnbindTexture( texmLatticeTmpRef                   );
    cudaUnbindTexture( texPolymerAndMonomerIsEvenAndOnXRef );
    cudaUnbindTexture( texMonomersSpezies_A_ThreadIdx      );
    cudaUnbindTexture( texMonomersSpezies_B_ThreadIdx      );

    cudaDestroyTextureObject( texSpeciesIndicesA );
    cudaDestroyTextureObject( texSpeciesIndicesB );
    texSpeciesIndicesA = 0;
    texSpeciesIndicesB = 0;

    //free memory on GPU
    cudaFree(mLatticeOut_device);
    cudaFree(mLatticeTmp_device);

    cudaFree(mPolymerSystem_device);
    cudaFree(MonoInfo_device);

    cudaFree(MonomersSpeziesIdx_A_device);
    cudaFree(MonomersSpeziesIdx_B_device);

    //free memory on CPU
    free(mPolymerSystem_host);
    free(MonoInfo_host);

    free(mLatticeOut_host);
    free(mLatticeTmp_host);

    free(MonomersSpeziesIdx_A_host);
    free(MonomersSpeziesIdx_B_host);

}
