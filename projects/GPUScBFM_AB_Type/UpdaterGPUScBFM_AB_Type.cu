/*
 * UpdaterGPUScBFM_AB_Type.cpp
 *
 *  Created on: 27.07.2017
 *      Author: Ron Dockhorn
 */


#include <cstdio>                           // printf
#include <cstdlib>                          // exit
#include <cstring>                          // memset
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <stdint.h>
#include <sstream>

#include "UpdaterGPUScBFM_AB_Type.h"

#define DEBUG_UPDATERGPUSCBFM_AB_TYPE 100


/* why 512??? Because 512==8^3 ??? but that would mean 8 possible values instead of
 * -4 to +4 which I saw being used ... */
__device__ __constant__ bool dpForbiddenBonds[512]; //false-allowed; true-forbidden

/**
 * These will be initialized to:
 *   DXTable_d = { -1,1,0,0,0,0 }
 *   DYTable_d = { 0,0,-1,1,0,0 }
 *   DZTable_d = { 0,0,0,0,-1,1 }
 * I.e. a table of three random directional 3D vectors \vec{dr} = (dx,dy,dz)
 */
__device__ __constant__ intCUDA DXTable_d[6]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
__device__ __constant__ intCUDA DYTable_d[6]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
__device__ __constant__ intCUDA DZTable_d[6]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z

/* will this really bring performance improvement? At least constant cache
 * might be as fast as register access when all threads in a warp access the
 * the same constant */
__device__ __constant__ uint32_t dcBoxXM1  ;  // mLattice size in X-1
__device__ __constant__ uint32_t dcBoxYM1  ;  // mLattice size in Y-1
__device__ __constant__ uint32_t dcBoxZM1  ;  // mLattice size in Z-1

__device__ __constant__ uint32_t dcBoxXLog2 ;  // mLattice shift in X
__device__ __constant__ uint32_t dcBoxXYLog2;  // mLattice shift in X*Y

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
/**
 * Contains the particles as well as a property tag for each:
 *   [ x0, y0, z0, p0, x1, y1, z1, p1, ... ]
 * The propertie tags p are bit packed:
 *                        8  7  6  5  4  3  2  1  0
 * +--------+--+--+--+--+--+--+--+--+--+--+--+--+--+
 * | unused |  |  |  |  |c |   nnr  |  dir   |move |
 * +--------+--+--+--+--+--+--+--+--+--+--+--+--+--+
 *  c   ... charged: 0 no, 1: yes
 *  nnr ... number of neighbors, this will get populated from LeMonADE's
 *          get get
 */
texture< intCUDA, cudaTextureType1D, cudaReadModeElementType > texPolymerAndMonomerIsEvenAndOnXRef;

cudaTextureObject_t texLatticeRefOut = 0;
cudaTextureObject_t texLatticeTmpRef = 0;

/**
 * These are arrays containing the monomer indices for the respective
 * species (sorted ascending). E.g. for AABABBA this would be:
 * texSpeciesIndicesA = { 0,1,3,6 }
 * texSpeciesIndicesB = { 1,4,5 }
 */
cudaTextureObject_t texSpeciesIndicesA = 0;
cudaTextureObject_t texSpeciesIndicesB = 0;



__device__ uint32_t hash( uint32_t a )
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

template< typename T >
__device__ __host__ bool isPowerOfTwo( T const & x )
{
    return ! ( x == 0 ) && ! ( x & ( x - 1 ) );
}

uint32_t UpdaterGPUScBFM_AB_Type::linearizeBoxVectorIndex
(
    uint32_t const & ix,
    uint32_t const & iy,
    uint32_t const & iz
)
{
    #ifdef NOMAGIC
        return ( ix % mBoxX ) +
               ( iy % mBoxY ) * mBoxX +
               ( iz % mBoxZ ) * mBoxX * mBoxY;
    #else
        assert( isPowerOfTwo( mBoxXM1 + 1 ) );
        assert( isPowerOfTwo( mBoxYM1 + 1 ) );
        assert( isPowerOfTwo( mBoxZM1 + 1 ) );
        return   ( ix & mBoxXM1 ) +
               ( ( iy & mBoxYM1 ) << mBoxXLog2  ) +
               ( ( iz & mBoxZM1 ) << mBoxXYLog2 );
    #endif
}

__device__ uint32_t linearizeBoxVectorIndex
(
    uint32_t const & ix,
    uint32_t const & iy,
    uint32_t const & iz
)
{
    #if DEBUG_UPDATERGPUSCBFM_AB_TYPE > 10
        assert( isPowerOfTwo( mBoxXM1 + 1 ) );
        assert( isPowerOfTwo( mBoxYM1 + 1 ) );
        assert( isPowerOfTwo( mBoxZM1 + 1 ) );
    #endif
    return   ( ix & dcBoxXM1 ) +
           ( ( iy & dcBoxYM1 ) << dcBoxXLog2  ) +
           ( ( iz & dcBoxZM1 ) << dcBoxXYLog2 );
}

__device__ inline bool checkLattice
(
    cudaTextureObject_t const texLattice,
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
#if 0 // defined( NOMAGIC ) // boh versions successfully tested :)
    /* positions after movement. Why 2 times dx ??? */
    uint32_t const x1 = ( x0 + dx + dx ) & dcBoxXM1;
    uint32_t const y1 = ( y0 + dy + dy ) & dcBoxYM1;
    uint32_t const z1 = ( z0 + dz + dz ) & dcBoxZM1;
    switch ( axis )
    {
        #define TMP_FETCH( x,y,z ) \
            tex1Dfetch< uint8_t >( texLattice, linearizeBoxVectorIndex(x,y,z) )
        case 0: //-+x
            test = TMP_FETCH( x1, y0 - 1, z0     ) |
                   TMP_FETCH( x1, y0    , z0     ) |
                   TMP_FETCH( x1, y0 + 1, z0     ) |
                   TMP_FETCH( x1, y0 - 1, z0 - 1 ) |
                   TMP_FETCH( x1, y0    , z0 - 1 ) |
                   TMP_FETCH( x1, y0 + 1, z0 - 1 ) |
                   TMP_FETCH( x1, y0 - 1, z0 + 1 ) |
                   TMP_FETCH( x1, y0    , z0 + 1 ) |
                   TMP_FETCH( x1, y0 + 1, z0 + 1 );
            break;
        case 1: //-+y
            test = TMP_FETCH( x0 - 1, y1, z0 - 1 ) |
                   TMP_FETCH( x0    , y1, z0 - 1 ) |
                   TMP_FETCH( x0 + 1, y1, z0 - 1 ) |
                   TMP_FETCH( x0 - 1, y1, z0     ) |
                   TMP_FETCH( x0    , y1, z0     ) |
                   TMP_FETCH( x0 + 1, y1, z0     ) |
                   TMP_FETCH( x0 - 1, y1, z0 + 1 ) |
                   TMP_FETCH( x0    , y1, z0 + 1 ) |
                   TMP_FETCH( x0 + 1, y1, z0 + 1 );
            break;
        case 2: //-+z
            test = TMP_FETCH( x0 - 1, y0 - 1, z1 ) |
                   TMP_FETCH( x0    , y0 - 1, z1 ) |
                   TMP_FETCH( x0 + 1, y0 - 1, z1 ) |
                   TMP_FETCH( x0 - 1, y0    , z1 ) |
                   TMP_FETCH( x0    , y0    , z1 ) |
                   TMP_FETCH( x0 + 1, y0    , z1 ) |
                   TMP_FETCH( x0 - 1, y0 + 1, z1 ) |
                   TMP_FETCH( x0    , y0 + 1, z1 ) |
                   TMP_FETCH( x0 + 1, y0 + 1, z1 );
            break;
        #undef TMP_FETCH
    }
#else
    uint32_t const x1     =   ( x0 + dx + dx ) & dcBoxXM1;
    uint32_t const y1     = ( ( y0 + dy + dy ) & dcBoxYM1 ) << dcBoxXLog2;
    uint32_t const z1     = ( ( z0 + dz + dz ) & dcBoxZM1 ) << dcBoxXYLog2;
    uint32_t const x0Abs  =   ( x0     ) & dcBoxXM1;
    uint32_t const x0PDX  =   ( x0 + 1 ) & dcBoxXM1;
    uint32_t const x0MDX  =   ( x0 - 1 ) & dcBoxXM1;
    uint32_t const y0Abs  = ( ( y0     ) & dcBoxYM1 ) << dcBoxXLog2;
    uint32_t const y0PDY  = ( ( y0 + 1 ) & dcBoxYM1 ) << dcBoxXLog2;
    uint32_t const y0MDY  = ( ( y0 - 1 ) & dcBoxYM1 ) << dcBoxXLog2;
    uint32_t const z0Abs  = ( ( z0     ) & dcBoxZM1 ) << dcBoxXYLog2;
    uint32_t const z0PDZ  = ( ( z0 + 1 ) & dcBoxZM1 ) << dcBoxXYLog2;
    uint32_t const z0MDZ  = ( ( z0 - 1 ) & dcBoxZM1 ) << dcBoxXYLog2;

    switch ( axis )
    {
        case 0: //-+x
            test = tex1Dfetch< uint8_t >( texLattice, x1 + y0MDY + z0Abs ) |
                   tex1Dfetch< uint8_t >( texLattice, x1 + y0Abs + z0Abs ) |
                   tex1Dfetch< uint8_t >( texLattice, x1 + y0PDY + z0Abs ) |
                   tex1Dfetch< uint8_t >( texLattice, x1 + y0MDY + z0MDZ ) |
                   tex1Dfetch< uint8_t >( texLattice, x1 + y0Abs + z0MDZ ) |
                   tex1Dfetch< uint8_t >( texLattice, x1 + y0PDY + z0MDZ ) |
                   tex1Dfetch< uint8_t >( texLattice, x1 + y0MDY + z0PDZ ) |
                   tex1Dfetch< uint8_t >( texLattice, x1 + y0Abs + z0PDZ ) |
                   tex1Dfetch< uint8_t >( texLattice, x1 + y0PDY + z0PDZ );
            break;
        case 1: //-+y
            test = tex1Dfetch< uint8_t >( texLattice, x0MDX + y1 + z0MDZ ) |
                   tex1Dfetch< uint8_t >( texLattice, x0Abs + y1 + z0MDZ ) |
                   tex1Dfetch< uint8_t >( texLattice, x0PDX + y1 + z0MDZ ) |
                   tex1Dfetch< uint8_t >( texLattice, x0MDX + y1 + z0Abs ) |
                   tex1Dfetch< uint8_t >( texLattice, x0Abs + y1 + z0Abs ) |
                   tex1Dfetch< uint8_t >( texLattice, x0PDX + y1 + z0Abs ) |
                   tex1Dfetch< uint8_t >( texLattice, x0MDX + y1 + z0PDZ ) |
                   tex1Dfetch< uint8_t >( texLattice, x0Abs + y1 + z0PDZ ) |
                   tex1Dfetch< uint8_t >( texLattice, x0PDX + y1 + z0PDZ );
            break;
        case 2: //-+z
            test = tex1Dfetch< uint8_t >( texLattice, x0MDX + y0MDY + z1 ) |
                   tex1Dfetch< uint8_t >( texLattice, x0Abs + y0MDY + z1 ) |
                   tex1Dfetch< uint8_t >( texLattice, x0PDX + y0MDY + z1 ) |
                   tex1Dfetch< uint8_t >( texLattice, x0MDX + y0Abs + z1 ) |
                   tex1Dfetch< uint8_t >( texLattice, x0Abs + y0Abs + z1 ) |
                   tex1Dfetch< uint8_t >( texLattice, x0PDX + y0Abs + z1 ) |
                   tex1Dfetch< uint8_t >( texLattice, x0MDX + y0PDY + z1 ) |
                   tex1Dfetch< uint8_t >( texLattice, x0Abs + y0PDY + z1 ) |
                   tex1Dfetch< uint8_t >( texLattice, x0PDX + y0PDY + z1 );
            break;
    }
#endif
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
 * Note: all of the three kernels do quite few work. They basically just fetch
 *       data, and check one condition and write out again. There isn't even
 *       a loop and most of the work seems to be boiler plate initialization
 *       code which could be cut if the kernels could be merged together.
 *       Why are there three kernels instead of just one
 *        -> for global synchronization
 */
__global__ void kernelSimulationScBFMCheckSpezies
(
    intCUDA           * const mPolymerSystem_d ,
    uint8_t           * const mLatticeTmp_d    ,
    MonoInfo          * const MonoInfo_d       ,
    cudaTextureObject_t const texSpeciesIndices,
    uint32_t            const nMonomers        ,
    uint32_t            const rn               ,
    cudaTextureObject_t const texLatticeRefOut
)
{
    int linId = blockIdx.x * blockDim.x + threadIdx.x;
    /* might be more readable to just return if the thread is masked ???
     * if ( ! ( linId < nMonomers ) )
     *     return;
     * I think it only works on newer CUDA versions ??? else the whole warp
     * might quit???
     */
    if ( linId < nMonomers )
    {
        // "select random monomer" ??? I don't see why this is random? texSpeciesIndices is not randomized!
        uint32_t const iMonomer   = tex1Dfetch< uint32_t >( texSpeciesIndices, linId );
        /* isn't this basically an array of structs where a struct of arrays
         * should be faster ??? */
        intCUDA  const x0         = tex1Dfetch( texPolymerAndMonomerIsEvenAndOnXRef, 4*iMonomer+0 );
        intCUDA  const y0         = tex1Dfetch( texPolymerAndMonomerIsEvenAndOnXRef, 4*iMonomer+1 );
        intCUDA  const z0         = tex1Dfetch( texPolymerAndMonomerIsEvenAndOnXRef, 4*iMonomer+2 );
        intCUDA  const properties = tex1Dfetch( texPolymerAndMonomerIsEvenAndOnXRef, 4*iMonomer+3 );

        //select random direction. Own implementation of an rng :S? But I think it at least# was initialized using the LeMonADE RNG ...
        uintCUDA const random_int = hash( hash( linId ) ^ rn ) % 6;

         //select random direction. !!! That table is kinda magic. there might be a better way ... E.g. using bitmasking. Also, what is with 0 in one direction ??? There is no way to e.g. get (0,1,-1) ... ???
         //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
        intCUDA const dx = DXTable_d[ random_int ];
        intCUDA const dy = DYTable_d[ random_int ];
        intCUDA const dz = DZTable_d[ random_int ];

#ifdef NONPERIODICITY
       /* check whether the new location of the particle would be inside the box
        * if the box is not periodic, if not, then don't move the particle */
        if ( ! ( 0 <= x0 + dx && x0 + dx < dcBoxXM1 &&
                 0 <= y0 + dy && y0 + dy < dcBoxYM1 &&
                 0 <= z0 + dz && z0 + dz < dcBoxZM1 ) )
        {
            return;
        }
#endif
        const unsigned nextNeigborSize = ( properties & 224 ) >> 5; // 224 = 0b1110 0000
        for ( unsigned u = 0; u < nextNeigborSize; ++u )
        {
            intCUDA const nN_X = tex1Dfetch( texPolymerAndMonomerIsEvenAndOnXRef, 4*MonoInfo_d[iMonomer].bondsMonomerIdx[u]+0 );
            intCUDA const nN_Y = tex1Dfetch( texPolymerAndMonomerIsEvenAndOnXRef, 4*MonoInfo_d[iMonomer].bondsMonomerIdx[u]+1 );
            intCUDA const nN_Z = tex1Dfetch( texPolymerAndMonomerIsEvenAndOnXRef, 4*MonoInfo_d[iMonomer].bondsMonomerIdx[u]+2 );
            if ( dpForbiddenBonds[ IdxBondArray_d( nN_X - x0 - dx, nN_Y - y0 - dy, nN_Z - z0 - dz ) ] )
                return;
        }

        if ( checkLattice( texLatticeRefOut, x0, y0, z0, dx, dy, dz, random_int >> 1 ) )
            return;

        // everything fits -> perform the move - add the information
        // possible move
        /* ??? can I simply & dcBoxXM1 ? this looks like something like
         * ( x0+dx ) % xmax is trying to be achieved. Using bitmasking for that
         * is only possible if dcBoxXM1+1 is a power of two ... */
        mPolymerSystem_d[ 4*iMonomer+3 ] = properties | ((random_int<<2)+1);
        mLatticeTmp_d[ linearizeBoxVectorIndex( x0+dx, y0+dy, z0+dz ) ] = 1;
    }
}

__global__ void kernelSimulationScBFMPerformSpecies
(
    intCUDA             * const mPolymerSystem_d ,
    uint8_t             * const mLattice_d       ,
    cudaTextureObject_t   const texSpeciesIndices,
    uint32_t              const nMonomers        ,
    cudaTextureObject_t   const texLatticeTmpRef
)
{
    int const linId = blockIdx.x * blockDim.x + threadIdx.x;
    if ( linId < nMonomers )
    {
        uint32_t const iMonomer   = tex1Dfetch< uint32_t >( texSpeciesIndices, linId );
        intCUDA  const properties = tex1Dfetch( texPolymerAndMonomerIsEvenAndOnXRef, 4*iMonomer+3 );
        if ( properties & 1 != 0 )    // possible move
        {
            intCUDA  const x0 = tex1Dfetch( texPolymerAndMonomerIsEvenAndOnXRef, 4*iMonomer+0 );
            intCUDA  const y0 = tex1Dfetch( texPolymerAndMonomerIsEvenAndOnXRef, 4*iMonomer+1 );
            intCUDA  const z0 = tex1Dfetch( texPolymerAndMonomerIsEvenAndOnXRef, 4*iMonomer+2 );
            uintCUDA const random_int = ( properties & 28 ) >> 2; // 28 == 0b11100

            intCUDA const dx = DXTable_d[ random_int ];
            intCUDA const dy = DYTable_d[ random_int ];
            intCUDA const dz = DZTable_d[ random_int ];

            if ( checkLattice( texLatticeTmpRef, x0, y0, z0, dx, dy, dz, random_int >> 1 ) )
                return;

            // everything fits -> perform the move - add the information
            //mPolymerSystem_d[ 4*iMonomer+0 ] = x0 + dx;
            //mPolymerSystem_d[ 4*iMonomer+1 ] = y0 + dy;
            //mPolymerSystem_d[ 4*iMonomer+2 ] = z0 + dz;
            mPolymerSystem_d[ 4*iMonomer+3 ] = properties | 2; // indicating allowed move
            mLattice_d[ linearizeBoxVectorIndex( x0+dx, y0+dy, z0+dz ) ] = 1;
            mLattice_d[ linearizeBoxVectorIndex( x0, y0, z0 ) ] = 0;
        }
    }
}

__global__ void kernelSimulationScBFMZeroArraySpecies
(
    intCUDA             * const mPolymerSystem_d ,
    uint8_t             * const mLatticeTmp_d    ,
    cudaTextureObject_t   const texSpeciesIndices,
    uint32_t              const nMonomers
)
{
    int linId = blockIdx.x * blockDim.x + threadIdx.x;
    if ( linId < nMonomers )
    {
        uint32_t const iMonomer = tex1Dfetch< uint32_t >( texSpeciesIndices, linId );
        intCUDA  const properties = tex1Dfetch( texPolymerAndMonomerIsEvenAndOnXRef, 4*iMonomer+3 );

        if ( ( properties & 3 ) != 0 )    //possible move
        {
            intCUDA const x0 = tex1Dfetch( texPolymerAndMonomerIsEvenAndOnXRef, 4*iMonomer+0 );
            intCUDA const y0 = tex1Dfetch( texPolymerAndMonomerIsEvenAndOnXRef, 4*iMonomer+1 );
            intCUDA const z0 = tex1Dfetch( texPolymerAndMonomerIsEvenAndOnXRef, 4*iMonomer+2 );

            //select random direction
            uintCUDA const random_int = ( properties & 28 ) >> 2;

            //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
            intCUDA const dx = DXTable_d[ random_int ];
            intCUDA const dy = DYTable_d[ random_int ];
            intCUDA const dz = DZTable_d[ random_int ];

            // possible move but not allowed
            if ( ( properties & 3 ) == 1 )
            {
                mLatticeTmp_d[ linearizeBoxVectorIndex( x0+dx, y0+dy, z0+dz ) ] = 0;
                mPolymerSystem_d[ 4*iMonomer+3 ] = properties & MASK5BITS; // delete the first 5 bits
            }
            else //allowed move with all circumstance
            {
                mPolymerSystem_d[ 4*iMonomer+0 ] = x0 + dx;
                mPolymerSystem_d[ 4*iMonomer+1 ] = y0 + dy;
                mPolymerSystem_d[ 4*iMonomer+2 ] = z0 + dz;
                mPolymerSystem_d[ 4*iMonomer+3 ] = properties & MASK5BITS; // delete the first 5 bits
                mLatticeTmp_d[ linearizeBoxVectorIndex( x0+dx, y0+dy, z0+dz ) ] = 0;
            }
            // everything fits -> perform the move - add the information
            //  mPolymerSystem_d[4*iMonomer+3] = properties & MASK5BITS; // delete the first 5 bits <- this comment was only for species B
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
    bool * tmpForbiddenBonds = (bool *) malloc(sizeof(bool)*512);
    uint nAllowedBonds = 0;
    for(int i = 0; i < 512; i++)
    {
        tmpForbiddenBonds[i] = false;
        tmpForbiddenBonds[i] = mForbiddenBonds[i];
        if ( ! tmpForbiddenBonds[i] )
            nAllowedBonds++;
    }
    std::cout << "used bonds in simulation: " << nAllowedBonds << " / 108 " << std::endl;
    if ( nAllowedBonds != 108 )
    {
        std::stringstream msg;
        msg << "Wrong bond-set! Expected 108 allowed bonds, but got " << nAllowedBonds << ". Exiting...\n";
        throw std::runtime_error( msg.str() );
    }
    CUDA_CHECK( cudaMemcpyToSymbol( dpForbiddenBonds, tmpForbiddenBonds, sizeof(bool)*512 ) );
    free(tmpForbiddenBonds);

    /* create a table mapping the random int to directions whereto move the
     * monomers */
    std::cout << "copy DXYZTable: " << std::endl;
    intCUDA tmp_DXTable[6] = { -1,1,  0,0,  0,0 };
    intCUDA tmp_DYTable[6] = {  0,0, -1,1,  0,0 };
    intCUDA tmp_DZTable[6] = {  0,0,  0,0, -1,1 };
    CUDA_CHECK( cudaMemcpyToSymbol( DXTable_d, tmp_DXTable, sizeof( intCUDA ) * 6 ) );
    CUDA_CHECK( cudaMemcpyToSymbol( DYTable_d, tmp_DYTable, sizeof( intCUDA ) * 6 ) );
    CUDA_CHECK( cudaMemcpyToSymbol( DZTable_d, tmp_DZTable, sizeof( intCUDA ) * 6 ) );

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

    /************************end: creating look-up for species*****************************************/

    /* calculate kernel config */
    /* ceilDiv better ??? */
    numblocksSpecies_A = (nMonomersSpeciesA-1)/NUMTHREADS+1;
    numblocksSpecies_B = (nMonomersSpeciesB-1)/NUMTHREADS+1;

    /****************************copy monomer informations ********************************************/
    mPolymerSystem_host =(intCUDA *) malloc((4*nAllMonomers+1)*sizeof(intCUDA));
    std::cout << "try to allocate : " << ((4*nAllMonomers+1)*sizeof(intCUDA)) << " bytes = " << ((4*nAllMonomers+1)*sizeof(intCUDA)/(1024.0)) << " kB = " << ((4*nAllMonomers+1)*sizeof(intCUDA)/(1024.0*1024.0)) << " MB coordinates on GPU " << std::endl;

    /* copy [ x0,y0,z0, x1 ... ] -> [ x0,y0,z0,p0, x1 ...]. Might be
     * an idea to use cudaMemcpy2D to transfer this strided array to GPU.
     * At least for copying back the results, see below, but for this the
     * property field actually will be set in the next few lines */
    CUDA_CHECK( cudaMalloc( (void **) &mPolymerSystem_device, ( 4*nAllMonomers+1 ) * sizeof( intCUDA ) ) );
    for ( uint32_t i =0; i < nAllMonomers; ++i )
    {
        mPolymerSystem_host[ 4*i+0 ] = (intCUDA) mPolymerSystem[ 3*i+0 ];
        mPolymerSystem_host[ 4*i+1 ] = (intCUDA) mPolymerSystem[ 3*i+1 ];
        mPolymerSystem_host[ 4*i+2 ] = (intCUDA) mPolymerSystem[ 3*i+2 ];
        mPolymerSystem_host[ 4*i+3 ] = 0;
    }

    // prepare and copy the connectivity matrix to GPU
    // the index on GPU starts at 0 and is one less than loaded
    int sizeMonoInfo = nAllMonomers * sizeof( MonoInfo );

    std::cout << "size of struct MonoInfo: " << sizeof(MonoInfo) << " bytes = " << (sizeof(MonoInfo)/(1024.0)) <<  "kB for one monomer connectivity " << std::endl;

    std::cout << "try to allocate : " << (sizeMonoInfo) << " bytes = " << (sizeMonoInfo/(1024.0)) <<  "kB = " << (sizeMonoInfo/(1024.0*1024.0)) <<  "MB for connectivity matrix on GPU " << std::endl;


    MonoInfo_host=(MonoInfo*) calloc(nAllMonomers,sizeof(MonoInfo));
    CUDA_CHECK(  cudaMalloc((void **) &MonoInfo_device, sizeMonoInfo));   // Allocate array of structure on device


    for ( uint32_t i = 0; i < nAllMonomers; ++i )
    {
        //MonoInfo_host[i].size = monosNNidx[i]->size;
        if((monosNNidx[i]->size) > 7)
        {
            std::cout << "this GPU-model allows max 7 next neighbors but size is " << (monosNNidx[i]->size) << ". Exiting..." << std::endl;
            throw std::runtime_error( "Limit of connectivity on GPU reached! Exiting...\n" );
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
    CUDA_CHECK( cudaMemcpy( MonoInfo_device, MonoInfo_host, sizeMonoInfo, cudaMemcpyHostToDevice ) );

    checkSystem();

    /* creating lattice */
    CUDA_CHECK( cudaMemcpyToSymbol( dcBoxXM1   , &mBoxXM1   , sizeof( mBoxXM1    ) ) );
    CUDA_CHECK( cudaMemcpyToSymbol( dcBoxYM1   , &mBoxYM1   , sizeof( mBoxYM1    ) ) );
    CUDA_CHECK( cudaMemcpyToSymbol( dcBoxZM1   , &mBoxZM1   , sizeof( mBoxZM1    ) ) );
    CUDA_CHECK( cudaMemcpyToSymbol( dcBoxXLog2 , &mBoxXLog2 , sizeof( mBoxXLog2  ) ) );
    CUDA_CHECK( cudaMemcpyToSymbol( dcBoxXYLog2, &mBoxXYLog2, sizeof( mBoxXYLog2 ) ) );

    mLatticeOut_host = (uint8_t *) malloc( mBoxX*mBoxY*mBoxZ*sizeof(uint8_t));
    mLatticeTmp_host = (uint8_t *) malloc( mBoxX*mBoxY*mBoxZ*sizeof(uint8_t));
    std::cout << "try to allocate : " << (mBoxX*mBoxY*mBoxZ*sizeof(uint8_t)) << " bytes = " << (mBoxX*mBoxY*mBoxZ*sizeof(uint8_t)/(1024.0*1024.0)) << " MB lattice on GPU " << std::endl;
    CUDA_CHECK( cudaMalloc( (void **) &mLatticeOut_device, mBoxX * mBoxY * mBoxZ * sizeof( *mLatticeOut_device ) ) );
    CUDA_CHECK( cudaMalloc( (void **) &mLatticeTmp_device, mBoxX * mBoxY * mBoxZ * sizeof( *mLatticeTmp_device ) ) );
    CUDA_CHECK( cudaMemset( mLatticeTmp_device, 0, mBoxX * mBoxY * mBoxZ * sizeof( *mLatticeTmp_device ) ) );

    std::memset( mLatticeOut_host, 0, mBoxX * mBoxY * mBoxZ * sizeof( *mLatticeOut_host ) );
    for ( int t = 0; t < nAllMonomers; ++t )
    {
        #ifdef USEZCURVE
            uint32_t xk = mPolymerSystem[ 3*t+0 ] & mBoxXM1;
            uint32_t yk = mPolymerSystem[ 3*t+1 ] & mBoxYM1;
            uint32_t zk = mPolymerSystem[ 3*t+2 ] & mBoxZM1;
            uint32_t inter3 = interleave3( xk/2 , yk/2, zk/2 );
            mLatticeOut_host[ ( ( mPolymerSystem_host[ 4*t+3 ] & 1 ) << 23 ) + inter3 ] = 1;
        #else
        mLatticeOut_host[ linearizeBoxVectorIndex( mPolymerSystem[ 3*t+0 ],
                                                   mPolymerSystem[ 3*t+1 ],
                                                   mPolymerSystem[ 3*t+2 ] ) ] = 1;
        #endif
    }
    CUDA_CHECK( cudaMemcpy( mLatticeOut_device, mLatticeOut_host, mBoxX * mBoxY * mBoxZ * sizeof( *mLatticeOut_host ), cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpy( mPolymerSystem_device, mPolymerSystem_host, ( 4*nAllMonomers+1 ) * sizeof( intCUDA ), cudaMemcpyHostToDevice ) );

    /* bind textures */
    cudaBindTexture( 0, texPolymerAndMonomerIsEvenAndOnXRef, mPolymerSystem_device, ( 4*nAllMonomers+1 ) * sizeof( intCUDA ) );

    /* new with texture object... they said it would be easier -.- */
    cudaResourceDesc resDescA;
    memset( &resDescA, 0, sizeof( resDescA ) );
    resDescA.resType                = cudaResourceTypeLinear;
    resDescA.res.linear.desc.f      = cudaChannelFormatKindUnsigned;
    resDescA.res.linear.desc.x      = 32; // bits per channel
    cudaResourceDesc resDescB = resDescA;
    cudaResourceDesc resDescRefOut = resDescA;
    resDescA.res.linear.devPtr      = MonomersSpeziesIdx_A_device;
    resDescA.res.linear.sizeInBytes = nMonomersSpeciesA * sizeof( uint32_t );
    resDescB.res.linear.devPtr      = MonomersSpeziesIdx_B_device;
    resDescB.res.linear.sizeInBytes = nMonomersSpeciesB * sizeof( uint32_t );

    cudaTextureDesc texDescROM;
    memset( &texDescROM, 0, sizeof( texDescROM ) );
    texDescROM.readMode = cudaReadModeElementType;

    /* the last three arguments are pointers to constants! */
    cudaCreateTextureObject( &texSpeciesIndicesA, &resDescA, &texDescROM, NULL );
    cudaCreateTextureObject( &texSpeciesIndicesB, &resDescB, &texDescROM, NULL );

    /* lattice textures */
    resDescRefOut.res.linear.desc.x = 8; // bits per channel
    resDescRefOut.res.linear.sizeInBytes = mBoxX*mBoxY*mBoxZ*sizeof(uint8_t);
    cudaResourceDesc resDescTmpRef = resDescRefOut;
    resDescRefOut.res.linear.devPtr = mLatticeOut_device;
    resDescTmpRef.res.linear.devPtr = mLatticeTmp_device;

    cudaCreateTextureObject( &texLatticeRefOut, &resDescRefOut, &texDescROM, NULL );
    cudaCreateTextureObject( &texLatticeTmpRef, &resDescTmpRef, &texDescROM, NULL );

    /* The memory transfer is this complex because int16_t needs to be
     * converted to int32_t if cudaInt == int16_t */
    if ( sizeof( *mPolymerSystem ) == sizeof( *mPolymerSystem_device ) )
    {
        std::cerr << "[" << __FILENAME__ << "::initialize] Can use cudaMemcpy2D (untested!)\n";
        throw std::runtime_error( "Untested. Uncomment if you work on it." );
        CUDA_CHECK( cudaMemcpy2D(
            mPolymerSystem       , 3 * sizeof( *mPolymerSystem ),
            mPolymerSystem_device, 4 * sizeof( *mPolymerSystem_device ),
            3 * sizeof( *mPolymerSystem ), nAllMonomers,
            cudaMemcpyDeviceToHost
        ) );
    }
    else
    {
        std::cerr << "[" << __FILENAME__ << "::initialize] Can't use cudaMemcpy2D"
            << " ( sizeof polymersystem, host: " << sizeof( *mPolymerSystem )
            << ", GPU: " << sizeof( *mPolymerSystem_device ) << ")\n";
        CUDA_CHECK( cudaMemcpy( mPolymerSystem_host, mPolymerSystem_device, ( 4*nAllMonomers+1 ) * sizeof( intCUDA ), cudaMemcpyDeviceToHost ) );
        for( uint32_t i = 0; i < nAllMonomers; ++i )
        {
            mPolymerSystem[ 3*i+0 ] = (int32_t) mPolymerSystem_host[ 4*i+0 ];
            mPolymerSystem[ 3*i+1 ] = (int32_t) mPolymerSystem_host[ 4*i+1 ];
            mPolymerSystem[ 3*i+2 ] = (int32_t) mPolymerSystem_host[ 4*i+2 ];
        }
    }

    std::cout << "check system before simulation: " << std::endl;
    checkSystem();
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

void UpdaterGPUScBFM_AB_Type::setNetworkIngredients( uint32_t numPEG, uint32_t numPEGArm, uint32_t numCL )
{
    nStars              = numPEG;    //number of Stars
    nMonomersPerStarArm = numPEGArm; //number OfMonomersPerStarArm
    nCrosslinker        = numCL;     //number of Crosslinker

    std::cout << "NumPEG on GPU         : " << nStars              << std::endl;
    std::cout << "NumPEGArmlength on GPU: " << nMonomersPerStarArm << std::endl;
    std::cout << "NumCrosslinker on GPU : " << nCrosslinker        << std::endl;

    //if (nMonomersPerStarArm != 29)
        //throw std::runtime_error("nMonomersPerStarArm should be 29! Exiting...\n");
    //if ((nMonomersPerStarArm%2) != 1)
        //    throw std::runtime_error("nMonomersPerStarArm should be an odd number! Exiting...\n");
}

void UpdaterGPUScBFM_AB_Type::setConnectivity(uint32_t monoidx1, uint32_t monoidx2)
{
    monosNNidx[monoidx1]->bondsMonomerIdx[monosNNidx[monoidx1]->size] = monoidx2;
    //monosNNidx[monoidx2]->bondsMonomerIdx[monosNNidx[monoidx2]->size] = monoidx1;

    monosNNidx[monoidx1]->size++;
    //monosNNidx[monoidx2]->size++;

    //if((monosNNidx[monoidx1]->size > MAX_CONNECTIVITY) || (monosNNidx[monoidx2]->size > MAX_CONNECTIVITY))
    if ( monosNNidx[monoidx1]->size > MAX_CONNECTIVITY )
        throw std::runtime_error("MAX_CONNECTIVITY  exceeded! Exiting...\n");
}

void UpdaterGPUScBFM_AB_Type::setLatticeSize
(
    uint32_t const boxX,
    uint32_t const boxY,
    uint32_t const boxZ
)
{
    mBoxX   = boxX;
    mBoxY   = boxY;
    mBoxZ   = boxZ;
    mBoxXM1 = boxX-1;
    mBoxYM1 = boxY-1;
    mBoxZM1 = boxZ-1;

    /* determine log2 for mBoxX and mBoxX * mBoxY to be used for bitshifting
     * the indice instead of multiplying ... WHY??? I don't think it is faster,
     * but much less readable */
    mBoxXLog2 = 0;
    uint32_t dummy = boxX;
    while ( dummy >>= 1 ) ++mBoxXLog2;
    mBoxXYLog2 = 0;
    dummy = boxX*boxY;
    while ( dummy >>= 1 ) ++mBoxXYLog2;

    std::cout
        << "use bit shift for boxX     : (1 << "<< mBoxXLog2  << " ) = "
        << ( 1 << mBoxXLog2  ) << " = " << mBoxX
        << "use bit shift for boxX*boxY: (1 << "<< mBoxXYLog2 << " ) = "
        << ( 1 << mBoxXYLog2 ) << " = " << mBoxX*boxY
        << std::endl;

    // check if shift is correct
    if ( boxX != ( 1 << mBoxXLog2 ) || boxX * boxY != ( 1 << mBoxXYLog2 ) )
        throw std::runtime_error( "Could not determine value for bit shift. Sure your box size is a power of 2? Exiting...\n" );

    //init lattice
    mLattice = new uint8_t[ mBoxX * mBoxY * mBoxZ ];
    std::memset( (void *) mLattice, 0, mBoxX * mBoxY * mBoxZ * sizeof( *mLattice ) );
}

void UpdaterGPUScBFM_AB_Type::populateLattice()
{
    std::memset( mLattice, 0, mBoxX * mBoxY * mBoxZ * sizeof( *mLattice ) );
    for ( size_t i = 0; i < nAllMonomers; ++i )
    {
        mLattice[ linearizeBoxVectorIndex( mPolymerSystem[3*i+0],
                                           mPolymerSystem[3*i+1],
                                           mPolymerSystem[3*i+2] ) ] = 1;
    }
}

/**
 * Checks for excluded volume condition and for correctness of all monomer bonds
 * Beware, it useses and thereby thrashes mLattice. Might be cleaner to declare
 * as const and malloc and free some temporary buffer, but the time ...
 * https://randomascii.wordpress.com/2014/12/10/hidden-costs-of-memory-allocation/
 * "In my tests, for sizes ranging from 8 MB to 32 MB, the cost for a new[]/delete[] pair averaged about 7.5 μs (microseconds), split into ~5.0 μs for the allocation and ~2.5 μs for the free."
 *  => ~40k cycles
 */
void UpdaterGPUScBFM_AB_Type::checkSystem()
{
    /**
     * Test for excluded volume by setting all lattice points and count the
     * toal lattice points occupied. If we have overlap this will be smaller
     * than calculated for zero overlap!
     * mPolymerSystem only stores the lower left front corner of the 2x2x2
     * monomer cube. Use that information to set all 8 cells in the lattice
     * to 'occupied'
     */
    /*
     Lattice is an array of size Box_X*Box_Y*Box_Z. PolymerSystem holds the monomer positions which I strongly guess are supposed to be in the range 0<=x<Box_X. If I see correctly, then this part checks for excluded volume by occupying a 2x2x2 cube for each monomer in Lattice and then counting the total occupied cells and compare it to the theoretical value of nMonomers * 8. But Lattice seems to be too small for that kinda usage! I.e. for two particles, one being at x=0 and the other being at x=Box_X-1 this test should return that the excluded volume condition is not met! Therefore the effective box size is actually (Box_X-1,Box_X-1,Box_Z-1) which in my opinion should be a bug ??? */
    std::memset( mLattice, 0, mBoxX * mBoxY * mBoxZ * sizeof( *mLattice ) );
    for ( int i = 0; i < nAllMonomers; ++i )
    {
        int32_t const & x = mPolymerSystem[ 3*i   ];
        int32_t const & y = mPolymerSystem[ 3*i+1 ];
        int32_t const & z = mPolymerSystem[ 3*i+2 ];
        /**
         * @verbatim
         *           ...+---+---+
         *     ...'''   | 6 | 7 |
         *    +---+---+ +---+---+    y
         *    | 2 | 3 | | 4 | 5 |    ^ z
         *    +---+---+ +---+---+    |/
         *    | 0 | 1 |   ...'''     +--> x
         *    +---+---+'''
         * @endverbatim
         */
        mLattice[ linearizeBoxVectorIndex( x  , y  , z   ) ] = 1; /* 0 */
        mLattice[ linearizeBoxVectorIndex( x+1, y  , z   ) ] = 1; /* 1 */
        mLattice[ linearizeBoxVectorIndex( x  , y+1, z   ) ] = 1; /* 2 */
        mLattice[ linearizeBoxVectorIndex( x+1, y+1, z   ) ] = 1; /* 3 */
        mLattice[ linearizeBoxVectorIndex( x  , y  , z+1 ) ] = 1; /* 4 */
        mLattice[ linearizeBoxVectorIndex( x+1, y  , z+1 ) ] = 1; /* 5 */
        mLattice[ linearizeBoxVectorIndex( x  , y+1, z+1 ) ] = 1; /* 6 */
        mLattice[ linearizeBoxVectorIndex( x+1, y+1, z+1 ) ] = 1; /* 7 */
    }
    /* check total occupied cells inside lattice to ensure that the above
     * transfer went without problems. Note that the number will be smaller
     * if some monomers overlap!
     * Could also simply reduce mLattice with +, I think, because it only
     * cotains 0 or 1 ??? */
    unsigned nOccupied = 0;
    for ( unsigned i = 0u; i < mBoxX * mBoxY * mBoxZ; ++i )
        nOccupied += mLattice[i] != 0;
    if ( ! ( nOccupied == nAllMonomers * 8 ) )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::~checkSystem" << "] "
            << "Occupation count in mLattice is wrong! Expected 8*nMonomers="
            << 8 * nAllMonomers << " occupied cells, but got " << nOccupied;
        throw std::runtime_error( msg.str() );
    }

    /**
     * Check bonds i.e. that |dx|<=3 and whether it is allowed by the given
     * bond set
     */
    for ( unsigned i = 0; i < nAllMonomers; ++i )
    for ( unsigned idxNN = 0; idxNN < monosNNidx[i]->size; ++idxNN )
    {
        /* calculate the bond vector between the neighbor and this particle
         * neighbor - particle = ( dx, dy, dz ) */
        int32_t const dx = mPolymerSystem[ 3*monosNNidx[i]->bondsMonomerIdx[idxNN]+0 ] - mPolymerSystem[ 3*i+0 ];
        int32_t const dy = mPolymerSystem[ 3*monosNNidx[i]->bondsMonomerIdx[idxNN]+1 ] - mPolymerSystem[ 3*i+1 ];
        int32_t const dz = mPolymerSystem[ 3*monosNNidx[i]->bondsMonomerIdx[idxNN]+2 ] - mPolymerSystem[ 3*i+2 ];

        int erroneousAxis = -1;
        if ( ! ( -3 <= dx && dx <= 3 ) ) erroneousAxis = 0;
        if ( ! ( -3 <= dy && dy <= 3 ) ) erroneousAxis = 1;
        if ( ! ( -3 <= dz && dz <= 3 ) ) erroneousAxis = 2;
        if ( erroneousAxis >= 0 || mForbiddenBonds[ IndexBondArray( dx, dy, dz ) ] )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::checkSystem] ";
            if ( erroneousAxis > 0 )
                msg << "Invalid " << 'X' + erroneousAxis << "Bond: ";
            if ( mForbiddenBonds[ IndexBondArray( dx, dy, dz ) ] )
                msg << "This particular bond is forbidden: ";
            msg << "(" << dx << "," << dy<< "," << dz << ") between monomer "
                << i+1 << " at (" << mPolymerSystem[3*i+0] << ","
                                  << mPolymerSystem[3*i+1] << ","
                                  << mPolymerSystem[3*i+2] << ") and monomer "
                << monosNNidx[i]->bondsMonomerIdx[idxNN]+1 << " at ("
                << mPolymerSystem[ 3*monosNNidx[i]->bondsMonomerIdx[idxNN]+0 ] << ","
                << mPolymerSystem[ 3*monosNNidx[i]->bondsMonomerIdx[idxNN]+1 ] << ","
                << mPolymerSystem[ 3*monosNNidx[i]->bondsMonomerIdx[idxNN]+2 ] << ")"
                << std::endl;
             throw std::runtime_error( msg.str() );
        }
    }
}

void UpdaterGPUScBFM_AB_Type::runSimulationOnGPU
(
    int32_t const nMonteCarloSteps
)
{
    std::clock_t const t0 = std::clock();

    /* run simulation */
    for ( int32_t iStep = 1; iStep <= nMonteCarloSteps; ++iStep )
    {
        /* one Monte-Carlo step */
        for ( uint32_t iSubStep = 0; iSubStep < 2; ++iSubStep )
        {
            switch ( randomNumbers.r250_rand32() % 2 )
            {
                case 0:  // run Spezies_A monomers
                    kernelSimulationScBFMCheckSpezies
                    <<< numblocksSpecies_A, NUMTHREADS >>>(
                        mPolymerSystem_device, mLatticeTmp_device,
                        MonoInfo_device, texSpeciesIndicesA,
                        nMonomersSpeciesA, randomNumbers.r250_rand32(),
                        texLatticeRefOut
                    );
                    kernelSimulationScBFMPerformSpecies
                    <<< numblocksSpecies_A, NUMTHREADS >>>(
                        mPolymerSystem_device, mLatticeOut_device,
                        texSpeciesIndicesA, nMonomersSpeciesA,
                        texLatticeTmpRef
                    );
                    kernelSimulationScBFMZeroArraySpecies
                    <<< numblocksSpecies_A, NUMTHREADS >>>(
                        mPolymerSystem_device, mLatticeTmp_device,
                        texSpeciesIndicesA, nMonomersSpeciesA
                    );
                    break;

                case 1: // run Spezies_B monomers
                    kernelSimulationScBFMCheckSpezies
                    <<< numblocksSpecies_B, NUMTHREADS >>>(
                        mPolymerSystem_device, mLatticeTmp_device,
                        MonoInfo_device, texSpeciesIndicesB,
                        nMonomersSpeciesB, randomNumbers.r250_rand32(),
                        texLatticeRefOut
                    );
                    kernelSimulationScBFMPerformSpecies
                    <<< numblocksSpecies_B, NUMTHREADS >>>(
                        mPolymerSystem_device, mLatticeOut_device,
                        texSpeciesIndicesB, nMonomersSpeciesB,
                        texLatticeTmpRef
                    );
                    kernelSimulationScBFMZeroArraySpecies
                    <<< numblocksSpecies_B, NUMTHREADS >>>(
                        mPolymerSystem_device, mLatticeTmp_device,
                        texSpeciesIndicesB, nMonomersSpeciesB
                    );
                    break;

                default: break;
            }
        }
    }

    /* all MCS are done- copy information back from GPU to host */
    CUDA_CHECK( cudaMemcpy( mLatticeTmp_host, mLatticeTmp_device, mBoxX * mBoxY * mBoxZ * sizeof( uint8_t ), cudaMemcpyDeviceToHost ) );

    unsigned nOccupied = 0;
    for ( unsigned i = 0u; i < mBoxX * mBoxY * mBoxZ; ++i )
        nOccupied += mLatticeTmp_host[i] != 0;
    if ( nOccupied != 0 )
    {
        std::stringstream msg;
        msg << "latticeTmp occupation (" << nOccupied << ") should be 0! Exiting ...\n";
        throw std::runtime_error( msg.str() );
    }

    /* why isn't this copied directly into mLattice ??? */
    CUDA_CHECK( cudaMemcpy( mLatticeOut_host, mLatticeOut_device, mBoxX * mBoxY * mBoxZ * sizeof( uint8_t ), cudaMemcpyDeviceToHost ) );
    for ( int i = 0; i < mBoxX * mBoxY * mBoxZ; ++i )
        mLattice[i] = mLatticeOut_host[i];

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

    /* copy into mPolymerSystem and drop the property tag while doing so.
     * would be easier and probably more efficient if mPolymerSystem_device/host
     * would be a struct of arrays instead of an array of structs !!! */
    CUDA_CHECK( cudaMemcpy( mPolymerSystem_host, mPolymerSystem_device, ( 4*nAllMonomers+1 ) * sizeof( intCUDA ), cudaMemcpyDeviceToHost ) );
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
    << "run time (GPU): " << nMonteCarloSteps << "\n"
    << "mcs = " << nMonteCarloSteps  << "  speed [performed monomer try and move/s] = MCS*N/t: "
    << nMonteCarloSteps * ( nAllMonomers / dt )  << "     runtime[s]:" << dt << std::endl;
}

void UpdaterGPUScBFM_AB_Type::cleanup()
{
    // copy information from GPU to Host
    CUDA_CHECK( cudaMemcpy( mLattice, mLatticeOut_device, mBoxX * mBoxY * mBoxZ * sizeof(uint8_t), cudaMemcpyDeviceToHost ) );
    CUDA_CHECK( cudaMemcpy( mPolymerSystem_host, mPolymerSystem_device, (4*nAllMonomers+1)*sizeof(intCUDA), cudaMemcpyDeviceToHost ) );
    for ( uint32_t i= 0 ; i < nAllMonomers; ++i )
    {
        mPolymerSystem[ 3*i+0 ] = (int32_t) mPolymerSystem_host[ 4*i+0 ];
        mPolymerSystem[ 3*i+1 ] = (int32_t) mPolymerSystem_host[ 4*i+1 ];
        mPolymerSystem[ 3*i+2 ] = (int32_t) mPolymerSystem_host[ 4*i+2 ];
    }

    checkSystem();

    // copy connectivity matrix back from device to host
    int sizeMonoInfo = nAllMonomers * sizeof(MonoInfo);
    CUDA_CHECK( cudaMemcpy(MonoInfo_host, MonoInfo_device, sizeMonoInfo, cudaMemcpyDeviceToHost));
    for (uint32_t i=0; i<nAllMonomers; i++)
    {
        //if(MonoInfo_host[i].size != monosNNidx[i]->size)
        if (  ( ( mPolymerSystem_host[4*i+3] & 224 ) >> 5 ) != monosNNidx[i]->size )
        {
            std::cout << "connectivity error after simulation run" << std::endl;
            std::cout << "mono:" << i << " vs " << (i) << std::endl;
            //cout << "numElements:" << MonoInfo_host[i].size << " vs " << monosNNidx[i]->size << endl;
            std::cout << "numElements:" << ((mPolymerSystem_host[4*i+3]&224)>>5) << " vs " << monosNNidx[i]->size << std::endl;

            throw std::runtime_error("Connectivity is corrupted! Maybe your Simulation is wrong! Exiting...\n");
        }
        for ( unsigned u = 0; u < MAX_CONNECTIVITY; ++u )
        {
            if ( MonoInfo_host[i].bondsMonomerIdx[u] != monosNNidx[i]->bondsMonomerIdx[u] )
            {
                std::cout << "connectivity error after simulation run" << std::endl;
                std::cout << "mono:" << i << " vs " << (i) << std::endl;

                std::cout << "bond["<< u << "]: " << MonoInfo_host[i].bondsMonomerIdx[u] << " vs " << monosNNidx[i]->bondsMonomerIdx[u] << std::endl;

                throw std::runtime_error("Connectivity is corrupted! Maybe your Simulation is wrong! Exiting...\n");
            }
        }
    }
    std::cout << "no errors in connectivity matrix after simulation run" << std::endl;

    checkSystem();

    //unbind texture reference to free resource
    cudaUnbindTexture( texPolymerAndMonomerIsEvenAndOnXRef );
    cudaDestroyTextureObject( texSpeciesIndicesA );
    cudaDestroyTextureObject( texSpeciesIndicesB );
    texSpeciesIndicesA = 0;
    texSpeciesIndicesB = 0;

    //free memory on GPU
    cudaFree( mLatticeOut_device          );
    cudaFree( mLatticeTmp_device          );
    cudaFree( mPolymerSystem_device       );
    cudaFree( MonoInfo_device             );
    cudaFree( MonomersSpeziesIdx_A_device );
    cudaFree( MonomersSpeziesIdx_B_device );

    //free memory on CPU
    free( mPolymerSystem_host       );
    free( MonoInfo_host             );
    free( mLatticeOut_host          );
    free( mLatticeTmp_host          );
    free( MonomersSpeziesIdx_A_host );
    free( MonomersSpeziesIdx_B_host );
}
