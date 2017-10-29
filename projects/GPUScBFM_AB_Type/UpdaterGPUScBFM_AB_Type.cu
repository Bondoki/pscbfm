/*
 * UpdaterGPUScBFM_AB_Type.cpp
 *
 *  Created on: 27.07.2017
 *      Author: Ron Dockhorn
 */

#include "UpdaterGPUScBFM_AB_Type.h"

#include <cstdio>                           // printf
#include <cstdlib>                          // exit
#include <cstring>                          // memset
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <stdint.h>
#include <sstream>

#include "cudacommon.hpp"

#define DEBUG_UPDATERGPUSCBFM_AB_TYPE 100


/* 512=8^3 for a range of bonds per direction of [-4,3] */
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
texture< intCUDA, cudaTextureType1D, cudaReadModeElementType > mPolymerSystem_texture;

cudaTextureObject_t texLatticeRefOut = 0;
cudaTextureObject_t texLatticeTmpRef = 0;



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

/**
 * Checks the 3x3 grid one in front of the new position in the direction of the
 * move given by axis.
 *
 * @verbatim
 *           ____________
 *         .'  .'  .'  .'|
 *        +---+---+---+  +     y
 *        | 6 | 7 | 8 |.'|     ^ z
 *        +---+---+---+  +     |/
 *        | 3/| 4/| 5 |.'|     +--> x
 *        +-/-+-/-+---+  +
 *   0 -> |+---+1/| 2 |.'  ^          ^
 *        /|/-/|/-+---+   /          / axis direction +z (axis = 0b101)
 *       / +-/-+         /  2 (*dz) /                              ++|
 *      +---+ /         /                                         /  +/-
 *      |/X |/         L                                        xyz
 *      +---+  <- X ... current position of the monomer
 * @endverbatim
 *
 * @param[in] axis +-x, +-y, +-z in that order from 0 to 5, or put in another
 *                 equivalent way: the lowest bit specifies +(1) or -(0) and the
 *                 Bit 2 and 1 specify the axis: 0b00=x, 0b01=y, 0b10=z
 * @return Returns true if any of that is occupied, i.e. if there
 *         would be a problem with the excluded volume condition.
 */
__device__ inline bool checkFront
(
    cudaTextureObject_t const & texLattice,
    intCUDA             const & x0        ,
    intCUDA             const & y0        ,
    intCUDA             const & z0        ,
    intCUDA             const & axis
)
{
    bool isOccupied = false;
#if 0
    #define TMP_FETCH( x,y,z ) \
        tex1Dfetch< uint8_t >( texLattice, linearizeBoxVectorIndex(x,y,z) )
    intCUDA const shift  = 4*(axis & 1)-2;
    intCUDA const iMove = axis >> 1;
    /* reduce branching by parameterizing the access axis, but that
     * makes the memory accesses more random again ???
     * for i0=0, i1=1, axis=z (same as in function doxygen ascii art)
     *    4 3 2
     *    5 0 1
     *    6 7 8
     */
    intCUDA r[3] = { x0, y0, z0 };
    r[ iMove ] += shift; isOccupied = TMP_FETCH( r[0], r[1], r[2] ); /* 0 */
    intCUDA i0 = iMove+1 >= 3 ? iMove+1-3 : iMove+1;
    intCUDA i1 = iMove+2 >= 3 ? iMove+2-3 : iMove+2;
    r[ i0 ]++; isOccupied |= TMP_FETCH( r[0], r[1], r[2] ); /* 1 */
    r[ i1 ]++; isOccupied |= TMP_FETCH( r[0], r[1], r[2] ); /* 2 */
    r[ i0 ]--; isOccupied |= TMP_FETCH( r[0], r[1], r[2] ); /* 3 */
    r[ i0 ]--; isOccupied |= TMP_FETCH( r[0], r[1], r[2] ); /* 4 */
    r[ i1 ]--; isOccupied |= TMP_FETCH( r[0], r[1], r[2] ); /* 5 */
    r[ i1 ]--; isOccupied |= TMP_FETCH( r[0], r[1], r[2] ); /* 6 */
    r[ i0 ]++; isOccupied |= TMP_FETCH( r[0], r[1], r[2] ); /* 7 */
    r[ i0 ]++; isOccupied |= TMP_FETCH( r[0], r[1], r[2] ); /* 8 */
    #undef TMP_FETCH
#elif 0 // defined( NOMAGIC )
    intCUDA const shift = 4*(axis & 1)-2;
    switch ( axis >> 1 )
    {
        #define TMP_FETCH( x,y,z ) \
            tex1Dfetch< uint8_t >( texLattice, linearizeBoxVectorIndex(x,y,z) )
        case 0: //-+x
        {
            uint32_t const x1 = x0 + shift;
            isOccupied = TMP_FETCH( x1, y0 - 1, z0     ) |
                         TMP_FETCH( x1, y0    , z0     ) |0
                         TMP_FETCH( x1, y0 + 1, z0     ) |
                         TMP_FETCH( x1, y0 - 1, z0 - 1 ) |
                         TMP_FETCH( x1, y0    , z0 - 1 ) |
                         TMP_FETCH( x1, y0 + 1, z0 - 1 ) |
                         TMP_FETCH( x1, y0 - 1, z0 + 1 ) |
                         TMP_FETCH( x1, y0    , z0 + 1 ) |
                         TMP_FETCH( x1, y0 + 1, z0 + 1 );
            break;
        }
        case 1: //-+y
        {
            uint32_t const y1 = y0 + shift;
            isOccupied = TMP_FETCH( x0 - 1, y1, z0 - 1 ) |
                         TMP_FETCH( x0    , y1, z0 - 1 ) |
                         TMP_FETCH( x0 + 1, y1, z0 - 1 ) |
                         TMP_FETCH( x0 - 1, y1, z0     ) |
                         TMP_FETCH( x0    , y1, z0     ) |
                         TMP_FETCH( x0 + 1, y1, z0     ) |
                         TMP_FETCH( x0 - 1, y1, z0 + 1 ) |
                         TMP_FETCH( x0    , y1, z0 + 1 ) |
                         TMP_FETCH( x0 + 1, y1, z0 + 1 );
            break;
        }
        case 2: //-+z
        {
            /**
             * @verbatim
             *   +---+---+---+  y
             *   | 6 | 7 | 8 |  ^ z
             *   +---+---+---+  |/
             *   | 3 | 4 | 5 |  +--> x
             *   +---+---+---+
             *   | 0 | 1 | 2 |
             *   +---+---+---+
             * @endverbatim
             */
            uint32_t const z1 = z0 + shift;
            isOccupied = TMP_FETCH( x0 - 1, y0 - 1, z1 ) | /* 0 */
                         TMP_FETCH( x0    , y0 - 1, z1 ) | /* 1 */
                         TMP_FETCH( x0 + 1, y0 - 1, z1 ) | /* 2 */
                         TMP_FETCH( x0 - 1, y0    , z1 ) | /* 3 */
                         TMP_FETCH( x0    , y0    , z1 ) | /* 4 */
                         TMP_FETCH( x0 + 1, y0    , z1 ) | /* 5 */
                         TMP_FETCH( x0 - 1, y0 + 1, z1 ) | /* 6 */
                         TMP_FETCH( x0    , y0 + 1, z1 ) | /* 7 */
                         TMP_FETCH( x0 + 1, y0 + 1, z1 );  /* 8 */
            break;
        }
        #undef TMP_FETCH
    }
#else
    uint32_t const x0Abs  =   ( x0     ) & dcBoxXM1;
    uint32_t const x0PDX  =   ( x0 + 1 ) & dcBoxXM1;
    uint32_t const x0MDX  =   ( x0 - 1 ) & dcBoxXM1;
    uint32_t const y0Abs  = ( ( y0     ) & dcBoxYM1 ) << dcBoxXLog2;
    uint32_t const y0PDY  = ( ( y0 + 1 ) & dcBoxYM1 ) << dcBoxXLog2;
    uint32_t const y0MDY  = ( ( y0 - 1 ) & dcBoxYM1 ) << dcBoxXLog2;
    uint32_t const z0Abs  = ( ( z0     ) & dcBoxZM1 ) << dcBoxXYLog2;
    uint32_t const z0PDZ  = ( ( z0 + 1 ) & dcBoxZM1 ) << dcBoxXYLog2;
    uint32_t const z0MDZ  = ( ( z0 - 1 ) & dcBoxZM1 ) << dcBoxXYLog2;

    intCUDA const dx = DXTable_d[ axis ];   // 2*axis-1
    intCUDA const dy = DYTable_d[ axis ];   // 2*(axis&1)-1
    intCUDA const dz = DZTable_d[ axis ];   // 2*(axis&1)-1
    switch ( axis >> 1 )
    {
        case 0: //-+x
        {
            uint32_t const x1 = ( x0 + 2*dx ) & dcBoxXM1;
            isOccupied =
                tex1Dfetch< uint8_t >( texLattice, x1 + y0MDY + z0Abs ) |
                tex1Dfetch< uint8_t >( texLattice, x1 + y0Abs + z0Abs ) |
                tex1Dfetch< uint8_t >( texLattice, x1 + y0PDY + z0Abs ) |
                tex1Dfetch< uint8_t >( texLattice, x1 + y0MDY + z0MDZ ) |
                tex1Dfetch< uint8_t >( texLattice, x1 + y0Abs + z0MDZ ) |
                tex1Dfetch< uint8_t >( texLattice, x1 + y0PDY + z0MDZ ) |
                tex1Dfetch< uint8_t >( texLattice, x1 + y0MDY + z0PDZ ) |
                tex1Dfetch< uint8_t >( texLattice, x1 + y0Abs + z0PDZ ) |
                tex1Dfetch< uint8_t >( texLattice, x1 + y0PDY + z0PDZ );
            break;
        }
        case 1: //-+y
        {
            uint32_t const y1 = ( ( y0 + 2*dy ) & dcBoxYM1 ) << dcBoxXLog2;
            isOccupied =
                tex1Dfetch< uint8_t >( texLattice, x0MDX + y1 + z0MDZ ) |
                tex1Dfetch< uint8_t >( texLattice, x0Abs + y1 + z0MDZ ) |
                tex1Dfetch< uint8_t >( texLattice, x0PDX + y1 + z0MDZ ) |
                tex1Dfetch< uint8_t >( texLattice, x0MDX + y1 + z0Abs ) |
                tex1Dfetch< uint8_t >( texLattice, x0Abs + y1 + z0Abs ) |
                tex1Dfetch< uint8_t >( texLattice, x0PDX + y1 + z0Abs ) |
                tex1Dfetch< uint8_t >( texLattice, x0MDX + y1 + z0PDZ ) |
                tex1Dfetch< uint8_t >( texLattice, x0Abs + y1 + z0PDZ ) |
                tex1Dfetch< uint8_t >( texLattice, x0PDX + y1 + z0PDZ );
            break;
        }
        case 2: //-+z
        {
            uint32_t const z1 = ( ( z0 + 2*dz ) & dcBoxZM1 ) << dcBoxXYLog2;
            isOccupied =
                tex1Dfetch< uint8_t >( texLattice, x0MDX + y0MDY + z1 ) |
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
    }
#endif
    return isOccupied;
}

__device__ __host__ uintCUDA linearizeBondVectorIndex
(
    intCUDA const x,
    intCUDA const y,
    intCUDA const z
)
{
    /* Just like for normal integers we clip the range to go more down than up
     * i.e. [-127 ,128] or in this case [-4,3] */
    assert( ( x & 7 ) == x );
    assert( ( y & 7 ) == y );
    assert( ( z & 7 ) == z );
    return   ( x & 7 /* 0b111 */ ) +
           ( ( y & 7 /* 0b111 */ ) << 3 ) +
           ( ( z & 7 /* 0b111 */ ) << 6 );
}

/**
 * Goes over all monomers of a species given specified by texSpeciesIndices
 * draws a random direction for them and checks whether that move is possible
 * with the box size and periodicity as well as the monomers at the target
 * location (excluded volume) and the new bond lengths to all neighbors.
 * If so, then the new position is set to 1 in dpLatticeTmp and encode the
 * possible movement direction in the property tag of the corresponding monomer
 * in dpPolymerSystem.
 *
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
__global__ void kernelSimulationScBFMCheckSpecies
(
    intCUDA           * const dpPolymerSystem  ,
    uint8_t           * const dpLatticeTmp     ,
    MonoInfo          * const dpMonoInfo       ,
    cudaTextureObject_t const texSpeciesIndices,
    uint32_t            const nMonomers        ,
    uint32_t            const rSeed            ,
    cudaTextureObject_t const texLatticeRefOut
)
{
    int const linId = blockIdx.x * blockDim.x + threadIdx.x;
    if ( linId >= nMonomers )
        return;

    // "select random monomer" ??? I don't see why this is random? texSpeciesIndices is not randomized!
    uint32_t const iMonomer   = tex1Dfetch< uint32_t >( texSpeciesIndices, linId );
    /* isn't this basically an array of structs where a struct of arrays
     * should be faster ??? */
    intCUDA  const x0         = tex1Dfetch( mPolymerSystem_texture, 4*iMonomer+0 );
    intCUDA  const y0         = tex1Dfetch( mPolymerSystem_texture, 4*iMonomer+1 );
    intCUDA  const z0         = tex1Dfetch( mPolymerSystem_texture, 4*iMonomer+2 );
    intCUDA  const properties = tex1Dfetch( mPolymerSystem_texture, 4*iMonomer+3 );

    //select random direction. Own implementation of an rng :S? But I think it at least# was initialized using the LeMonADE RNG ...
    uintCUDA const direction = hash( hash( linId ) ^ rSeed ) % 6;

     /* select random direction. Do this with bitmasking instead of lookup ??? */
    intCUDA const dx = DXTable_d[ direction ];
    intCUDA const dy = DYTable_d[ direction ];
    intCUDA const dz = DZTable_d[ direction ];

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
    /* check whether the new position would result in invalid bonds
     * between this monomer and its neighbors */
    unsigned const nNeighbors = ( properties & 224 ) >> 5; // 224 = 0b1110 0000
    for ( unsigned iNeighbor = 0; iNeighbor < nNeighbors; ++iNeighbor )
    {
        intCUDA const nN_X = tex1Dfetch( mPolymerSystem_texture, 4*dpMonoInfo[ iMonomer ].bondsMonomerIdx[ iNeighbor ]+0 );
        intCUDA const nN_Y = tex1Dfetch( mPolymerSystem_texture, 4*dpMonoInfo[ iMonomer ].bondsMonomerIdx[ iNeighbor ]+1 );
        intCUDA const nN_Z = tex1Dfetch( mPolymerSystem_texture, 4*dpMonoInfo[ iMonomer ].bondsMonomerIdx[ iNeighbor ]+2 );
        if ( dpForbiddenBonds[ linearizeBondVectorIndex( nN_X - x0 - dx, nN_Y - y0 - dy, nN_Z - z0 - dz ) ] )
            return;
    }

    if ( checkFront( texLatticeRefOut, x0, y0, z0, direction ) )
        return;

    // everything fits -> perform the move - add the information
    // possible move
    /* ??? can I simply & dcBoxXM1 ? this looks like something like
     * ( x0+dx ) % xmax is trying to be achieved. Using bitmasking for that
     * is only possible if dcBoxXM1+1 is a power of two ... */
    /* can I do this ??? dpPolymerSystem is the device pointer to the read-only
     * texture used above. Won't this result in read-after-write race-conditions?
     * Then again the written / changed bits are never used in the above code ... */
    dpPolymerSystem[ 4*iMonomer+3 ] = properties | ( ( direction << 2 ) + 1 );
    dpLatticeTmp[ linearizeBoxVectorIndex( x0+dx, y0+dy, z0+dz ) ] = 1;
}

__global__ void kernelSimulationScBFMPerformSpecies
(
    intCUDA             * const dpPolymerSystem ,
    uint8_t             * const dpLattice       ,
    cudaTextureObject_t   const texSpeciesIndices,
    uint32_t              const nMonomers        ,
    cudaTextureObject_t   const texLatticeTmpRef
)
{
    int const linId = blockIdx.x * blockDim.x + threadIdx.x;
    if ( linId >= nMonomers )
        return;

    uint32_t const iMonomer   = tex1Dfetch< uint32_t >( texSpeciesIndices, linId );
    intCUDA  const properties = tex1Dfetch( mPolymerSystem_texture, 4*iMonomer+3 );
    if ( ( properties & 1 ) == 0 )    // impossible move
        return;
    intCUDA  const x0 = tex1Dfetch( mPolymerSystem_texture, 4*iMonomer+0 );
    intCUDA  const y0 = tex1Dfetch( mPolymerSystem_texture, 4*iMonomer+1 );
    intCUDA  const z0 = tex1Dfetch( mPolymerSystem_texture, 4*iMonomer+2 );
    uintCUDA const direction = ( properties & 28 ) >> 2; // 28 == 0b11100

    intCUDA const dx = DXTable_d[ direction ];
    intCUDA const dy = DYTable_d[ direction ];
    intCUDA const dz = DZTable_d[ direction ];

    if ( checkFront( texLatticeTmpRef, x0, y0, z0, direction ) )
        return;

    // everything fits -> perform the move - add the information
    //dpPolymerSystem[ 4*iMonomer+0 ] = x0 + dx;
    //dpPolymerSystem[ 4*iMonomer+1 ] = y0 + dy;
    //dpPolymerSystem[ 4*iMonomer+2 ] = z0 + dz;
    dpPolymerSystem[ 4*iMonomer+3 ] = properties | 2; // indicating allowed move
    dpLattice[ linearizeBoxVectorIndex( x0+dx, y0+dy, z0+dz ) ] = 1;
    dpLattice[ linearizeBoxVectorIndex( x0, y0, z0 ) ] = 0;
}

__global__ void kernelSimulationScBFMZeroArraySpecies
(
    intCUDA             * const dpPolymerSystem ,
    uint8_t             * const dpLatticeTmp    ,
    cudaTextureObject_t   const texSpeciesIndices,
    uint32_t              const nMonomers
)
{
    int const linId = blockIdx.x * blockDim.x + threadIdx.x;
    if ( linId >= nMonomers )
        return;

    uint32_t const iMonomer = tex1Dfetch< uint32_t >( texSpeciesIndices, linId );
    intCUDA  const properties = tex1Dfetch( mPolymerSystem_texture, 4*iMonomer+3 );

    if ( ( properties & 3 ) == 0 )    // impossible move
        return;

    intCUDA const x0 = tex1Dfetch( mPolymerSystem_texture, 4*iMonomer+0 );
    intCUDA const y0 = tex1Dfetch( mPolymerSystem_texture, 4*iMonomer+1 );
    intCUDA const z0 = tex1Dfetch( mPolymerSystem_texture, 4*iMonomer+2 );

    //select random direction
    uintCUDA const direction = ( properties & 28 ) >> 2;

    //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
    intCUDA const dx = DXTable_d[ direction ];
    intCUDA const dy = DYTable_d[ direction ];
    intCUDA const dz = DZTable_d[ direction ];

    // possible move but not allowed
    if ( ( properties & 3 ) == 1 )
    {
        dpLatticeTmp[ linearizeBoxVectorIndex( x0+dx, y0+dy, z0+dz ) ] = 0;
        dpPolymerSystem[ 4*iMonomer+3 ] = properties & MASK5BITS; // delete the first 5 bits
    }
    else //allowed move with all circumstance
    {
        dpPolymerSystem[ 4*iMonomer+0 ] = x0 + dx;
        dpPolymerSystem[ 4*iMonomer+1 ] = y0 + dy;
        dpPolymerSystem[ 4*iMonomer+2 ] = z0 + dz;
        dpPolymerSystem[ 4*iMonomer+3 ] = properties & MASK5BITS; // delete the first 5 bits
        dpLatticeTmp[ linearizeBoxVectorIndex( x0+dx, y0+dy, z0+dz ) ] = 0;
    }
    // everything fits -> perform the move - add the information
    //  dpPolymerSystem_d[4*iMonomer+3] = properties & MASK5BITS; // delete the first 5 bits <- this comment was only for species B
}

UpdaterGPUScBFM_AB_Type::~UpdaterGPUScBFM_AB_Type()
{
    if ( mLattice         != NULL ){ delete[] mLattice        ; mLattice         = NULL; }
    if ( mPolymerSystem   != NULL ){ delete[] mPolymerSystem  ; mPolymerSystem   = NULL; }
    if ( mAttributeSystem != NULL ){ delete[] mAttributeSystem; mAttributeSystem = NULL; }
    if ( mNeighbors       != NULL ){ delete[] mNeighbors      ; mNeighbors       = NULL; }
    if ( mMonomerIdsA     != NULL ){ delete[] mMonomerIdsA    ; mMonomerIdsA     = NULL; }
    if ( mMonomerIdsB     != NULL ){ delete[] mMonomerIdsB    ; mMonomerIdsB     = NULL; }
}

void UpdaterGPUScBFM_AB_Type::initialize( int iGpuToUse )
{
    int nGpus;
    getCudaDeviceProperties( NULL, &nGpus, true /* pritn GPU information */ );
    if ( iGpuToUse >= nGpus )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::initialize] "
            << "GPU with ID " << iGpuToUse << " not present. "
            << "Only " << nGpus << " GPUs are available.\n";
        throw std::invalid_argument( msg.str() );
    }
    CUDA_CHECK( cudaSetDevice( iGpuToUse ));

    /* create the BondTable and copy it to constant memory */
    bool * tmpForbiddenBonds = (bool*) malloc( sizeof( bool ) * 512 );
    unsigned nAllowedBonds = 0;
    for ( int i = 0; i < 512; ++i )
        if ( ! ( tmpForbiddenBonds[i] = mForbiddenBonds[i] ) )
            ++nAllowedBonds;
    if ( nAllowedBonds != 108 )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::initialize] "
            << "Wrong bond-set! Expected 108 allowed bonds, but got " << nAllowedBonds << "\n";
        throw std::runtime_error( msg.str() );
    }
    CUDA_CHECK( cudaMemcpyToSymbol( dpForbiddenBonds, tmpForbiddenBonds, sizeof(bool)*512 ) );
    free( tmpForbiddenBonds );

    /* create a table mapping the random int to directions whereto move the monomers */
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

    mMonomerIdsA = new MirroredTexture< uint32_t >( nMonomersSpeciesA );
    mMonomerIdsB = new MirroredTexture< uint32_t >( nMonomersSpeciesB );

    /* sort monomers (their indices) into corresponding species array  */
    uint32_t nMonomersWrittenA = 0;
    uint32_t nMonomersWrittenB = 0;
    for ( uint32_t i = 0; i < nAllMonomers; ++i )
    {
        if ( pMonomerSpecies[i] == 1 )
            mMonomerIdsA->host[ nMonomersWrittenA++ ] = i;
        else if ( pMonomerSpecies[i] == 2 )
            mMonomerIdsB->host[ nMonomersWrittenB++ ] = i;
    }
    if ( nMonomersSpeciesA != nMonomersWrittenA )
        throw std::runtime_error( "Number of monomers copied for species A does not add up!" );
    if ( nMonomersSpeciesB != nMonomersWrittenB )
        throw std::runtime_error( "Number of monomers copied for species B does not add up!" );
     mMonomerIdsA->push();
     mMonomerIdsB->push();

    CUDA_CHECK( cudaMalloc( (void **) &mPolymerSystem_device, ( 4*nAllMonomers+1 ) * sizeof( intCUDA ) ) );

    // prepare and copy the connectivity matrix to GPU
    // the index on GPU starts at 0 and is one less than loaded
    int sizeMonoInfo = nAllMonomers * sizeof( MonoInfo );
    std::cout
        << "size of struct MonoInfo: " << sizeof(MonoInfo)
        << " bytes = " << (sizeof(MonoInfo)/(1024.0))
        <<  "kB for one monomer connectivity " << std::endl;
    std::cout << "try to allocate : " << (sizeMonoInfo) << " bytes = "
        << (sizeMonoInfo/(1024.0)) <<  "kB = " << (sizeMonoInfo/(1024.0*1024.0))
        <<  "MB for connectivity matrix on GPU " << std::endl;


    MonoInfo_host=(MonoInfo*) calloc(nAllMonomers,sizeof(MonoInfo));
    CUDA_CHECK( cudaMalloc((void **) &MonoInfo_device, sizeMonoInfo) );   // Allocate array of structure on device

    /* add property tags for each monomer with number of neighbor information */
    for ( uint32_t i = 0; i < nAllMonomers; ++i )
    {
        if ( mNeighbors[i].size > 7)
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::initialize] "
                << "This implementation allows max. 7 neighbors per monomer, "
                << "but monomer " << i << " has " << mNeighbors[i].size << "\n";
            throw std::invalid_argument( msg.str() );
        }
        mPolymerSystem[ 4*i+3 ] |= ( (intCUDA) mNeighbors[i].size ) << 5;
        for ( unsigned u = 0; u < MAX_CONNECTIVITY; ++u )
            MonoInfo_host[i].bondsMonomerIdx[u] = mNeighbors[i].bondsMonomerIdx[u];
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
            uint32_t xk = mPolymerSystem[ 4*t+0 ] & mBoxXM1;
            uint32_t yk = mPolymerSystem[ 4*t+1 ] & mBoxYM1;
            uint32_t zk = mPolymerSystem[ 4*t+2 ] & mBoxZM1;
            uint32_t inter3 = interleave3( xk/2 , yk/2, zk/2 );
            mLatticeOut_host[ ( ( mPolymerSystem_host[ 4*t+3 ] & 1 ) << 23 ) + inter3 ] = 1;
        #else
        mLatticeOut_host[ linearizeBoxVectorIndex( mPolymerSystem[ 4*t+0 ],
                                                   mPolymerSystem[ 4*t+1 ],
                                                   mPolymerSystem[ 4*t+2 ] ) ] = 1;
        #endif
    }
    CUDA_CHECK( cudaMemcpy( mLatticeOut_device, mLatticeOut_host, mBoxX * mBoxY * mBoxZ * sizeof( *mLatticeOut_host ), cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpy( mPolymerSystem_device, mPolymerSystem, ( 4*nAllMonomers+1 ) * sizeof( intCUDA ), cudaMemcpyHostToDevice ) );

    /* bind textures */
    cudaBindTexture( 0, mPolymerSystem_texture, mPolymerSystem_device, ( 4*nAllMonomers+1 ) * sizeof( intCUDA ) );

    /* new with texture object... they said it would be easier -.- */
    cudaResourceDesc resDescA;
    memset( &resDescA, 0, sizeof( resDescA ) );
    resDescA.resType                = cudaResourceTypeLinear;
    resDescA.res.linear.desc.f      = cudaChannelFormatKindUnsigned;
    resDescA.res.linear.desc.x      = 32; // bits per channel
    cudaResourceDesc resDescRefOut = resDescA;

    cudaTextureDesc texDescROM;
    memset( &texDescROM, 0, sizeof( texDescROM ) );
    texDescROM.readMode = cudaReadModeElementType;

    /* lattice textures */
    resDescRefOut.res.linear.desc.x = 8; // bits per channel
    resDescRefOut.res.linear.sizeInBytes = mBoxX*mBoxY*mBoxZ*sizeof(uint8_t);
    cudaResourceDesc resDescTmpRef = resDescRefOut;
    resDescRefOut.res.linear.devPtr = mLatticeOut_device;
    resDescTmpRef.res.linear.devPtr = mLatticeTmp_device;

    cudaCreateTextureObject( &texLatticeRefOut, &resDescRefOut, &texDescROM, NULL );
    cudaCreateTextureObject( &texLatticeTmpRef, &resDescTmpRef, &texDescROM, NULL );

    std::cerr << "[" << __FILENAME__ << "::initialize] Can't use cudaMemcpy2D"
        << " ( sizeof polymersystem, host: " << sizeof( *mPolymerSystem )
        << ", GPU: " << sizeof( *mPolymerSystem_device ) << ")\n";
    CUDA_CHECK( cudaMemcpy( mPolymerSystem, mPolymerSystem_device, ( 4*nAllMonomers+1 ) * sizeof( intCUDA ), cudaMemcpyDeviceToHost ) );

    checkSystem();
}


void UpdaterGPUScBFM_AB_Type::copyBondSet
( int dx, int dy, int dz, bool bondForbidden )
{
    mForbiddenBonds[ linearizeBondVectorIndex(dx,dy,dz) ] = bondForbidden;
}

void UpdaterGPUScBFM_AB_Type::setNrOfAllMonomers( uint32_t const rnAllMonomers )
{
    if ( this->nAllMonomers != 0 || mAttributeSystem != NULL ||
         mPolymerSystem != NULL || mNeighbors != NULL )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setNrOfAllMonomers] "
            << "Number of Monomers already set to " << nAllMonomers << "!\n"
            << "Or some arrays were already allocated "
            << "(mAttributeSystem=" << (void*) mAttributeSystem
            << ", mPolymerSystem" << (void*) mPolymerSystem
            << ", mNeighbors" << (void*) mNeighbors << ")\n";
        throw std::runtime_error( msg.str() );
    }

    this->nAllMonomers = rnAllMonomers;
    mAttributeSystem = new int32_t[ nAllMonomers ];
    mPolymerSystem   = new intCUDA[ nAllMonomers*4+1 ];
    mNeighbors       = new MonoNNIndex[ nAllMonomers ];
    std::memset( mNeighbors, 0, sizeof( mNeighbors[0] ) * nAllMonomers );
}

void UpdaterGPUScBFM_AB_Type::setPeriodicity
(
    bool const isPeriodicX,
    bool const isPeriodicY,
    bool const isPeriodicZ
)
{
    /* Compare inputs to hardcoded values. No ability yet to adjust dynamically */
    std::stringstream msg;
    msg << "[" << __FILENAME__ << "::setPeriodicity" << "] "
        << "Simulation is intended to use completely "
    #ifdef NONPERIODICITY
        << "non-"
    #endif
        << "periodic boundary conditions, but setPeriodicity was called with "
        << "(" << isPeriodicX << "," << isPeriodicY << "," << isPeriodicZ << ")\n";

#ifdef NONPERIODICITY
    if ( isPeriodicX || isPeriodicY || isPeriodicZ )
#else
    if ( ! isPeriodicX || ! isPeriodicY || ! isPeriodicZ )
#endif
        throw std::invalid_argument( msg.str() );
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

void UpdaterGPUScBFM_AB_Type::setMonomerCoordinates
(
    uint32_t const i,
    int32_t  const x,
    int32_t  const y,
    int32_t  const z
)
{
#if DEBUG_UPDATERGPUSCBFM_AB_TYPE > 1
    /* can I apply periodic modularity here to allow the full range ??? */
    if ( ! inRange< decltype( mPolymerSystem[0] ) >(x) ||
         ! inRange< decltype( mPolymerSystem[0] ) >(y) ||
         ! inRange< decltype( mPolymerSystem[0] ) >(z)    )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setMonomerCoordinates" << "] "
            << "One or more of the given coordinates "
            << "(" << x << "," << y << "," << z << ") "
            << "is larger than the internal integer data type for "
            << "representing positions allow! (" << std::numeric_limits< intCUDA >::min()
            << " <= size <= " << std::numeric_limits< intCUDA >::max() << ")";
        throw std::invalid_argument( msg.str() );
    }
#endif
    mPolymerSystem[ 4*i+0 ] = x;
    mPolymerSystem[ 4*i+1 ] = y;
    mPolymerSystem[ 4*i+2 ] = z;

    if ( mPolymerSystem == NULL )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setMonomerCoordinates" << "] "
            << "mPolymerSystem is not allocated. You need to call "
            << "setNrOfAllMonomers before calling setMonomerCoordinates!\n";
        throw std::invalid_argument( msg.str() );
    }
}

int32_t UpdaterGPUScBFM_AB_Type::getMonomerPositionInX( uint32_t i ){ return mPolymerSystem[ 4*i+0 ]; }
int32_t UpdaterGPUScBFM_AB_Type::getMonomerPositionInY( uint32_t i ){ return mPolymerSystem[ 4*i+1 ]; }
int32_t UpdaterGPUScBFM_AB_Type::getMonomerPositionInZ( uint32_t i ){ return mPolymerSystem[ 4*i+2 ]; }

void UpdaterGPUScBFM_AB_Type::setConnectivity
(
    uint32_t const iMonomer1,
    uint32_t const iMonomer2
)
{
    /* the commented parts are correct, but basically redundant, because
     * the bonds are a non-directional graph */
    mNeighbors[ iMonomer1 ].bondsMonomerIdx[ mNeighbors[ iMonomer1 ].size ] = iMonomer2;
    //mNeighbors[ iMonomer2 ].bondsMonomerIdx[ mNeighbors[ iMonomer2 ].size ] = iMonomer1;

    ++mNeighbors[ iMonomer1 ].size;
    //mNeighbors[ iMonomer2 ].size++;

    //if((mNeighbors[ iMonomer1 ].size > MAX_CONNECTIVITY) || (mNeighbors[ iMonomer2 ].size > MAX_CONNECTIVITY))
    if ( mNeighbors[ iMonomer1 ].size > MAX_CONNECTIVITY )
        throw std::runtime_error("MAX_CONNECTIVITY  exceeded! Exiting...\n");
}

void UpdaterGPUScBFM_AB_Type::setLatticeSize
(
    uint32_t const boxX,
    uint32_t const boxY,
    uint32_t const boxZ
)
{
    if ( mBoxX == boxX && mBoxY == boxY && mBoxZ == boxZ )
        return;

    if ( ! ( inRange< intCUDA >( boxX ) &&
             inRange< intCUDA >( boxY ) &&
             inRange< intCUDA >( boxZ )    ) )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setLatticeSize" << "] "
            << "The box size (" << boxX << "," << boxY << "," << boxZ
            << ") is larger than the internal integer data type for "
            << "representing positions allow! (" << std::numeric_limits< intCUDA >::min()
            << " <= size <= " << std::numeric_limits< intCUDA >::max() << ")";
        throw std::invalid_argument( msg.str() );
    }

    mBoxX   = boxX;
    mBoxY   = boxY;
    mBoxZ   = boxZ;
    mBoxXM1 = boxX-1;
    mBoxYM1 = boxY-1;
    mBoxZM1 = boxZ-1;

    /* determine log2 for mBoxX and mBoxX * mBoxY to be used for bitshifting
     * the indice instead of multiplying ... WHY??? I don't think it is faster,
     * but much less readable */
    mBoxXLog2  = 0; uint32_t dummy = mBoxX; while ( dummy >>= 1 ) ++mBoxXLog2;
    mBoxXYLog2 = 0; dummy = mBoxX*mBoxY;    while ( dummy >>= 1 ) ++mBoxXYLog2;
    if ( mBoxX != ( 1 << mBoxXLog2 ) || mBoxX * boxY != ( 1 << mBoxXYLog2 ) )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setLatticeSize" << "] "
            << "Could not determine value for bit shift. "
            << "Check whether the box size is a power of 2! ( "
            << "boxX=" << mBoxX << " =? 2^" << mBoxXLog2 << " = " << ( 1 << mBoxXLog2 )
            << ", boxX*boY=" << mBoxX * mBoxY << " =? 2^" << mBoxXYLog2
            << " = " << ( 1 << mBoxXYLog2 ) << " )\n";
        throw std::runtime_error( msg.str() );
    }

    if ( mLattice != NULL )
        delete[] mLattice;
    mLattice = new uint8_t[ mBoxX * mBoxY * mBoxZ ];
    std::memset( (void *) mLattice, 0, mBoxX * mBoxY * mBoxZ * sizeof( *mLattice ) );
}

void UpdaterGPUScBFM_AB_Type::populateLattice()
{
    std::memset( mLattice, 0, mBoxX * mBoxY * mBoxZ * sizeof( *mLattice ) );
    for ( size_t i = 0; i < nAllMonomers; ++i )
    {
        mLattice[ linearizeBoxVectorIndex( mPolymerSystem[ 4*i+0 ],
                                           mPolymerSystem[ 4*i+1 ],
                                           mPolymerSystem[ 4*i+2 ] ) ] = 1;
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
    if ( mLattice == NULL || mPolymerSystem == NULL )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::checkSystem" << "] "
            << "mPolymerSystem or mLattice is not allocated. You need to call "
            << "setNrOfAllMonomers and initialize before calling checkSystem!\n";
        throw std::invalid_argument( msg.str() );
    }

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
        int32_t const & x = mPolymerSystem[ 4*i   ];
        int32_t const & y = mPolymerSystem[ 4*i+1 ];
        int32_t const & z = mPolymerSystem[ 4*i+2 ];
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
    for ( unsigned iNeighbor = 0; iNeighbor < mNeighbors[i].size; ++iNeighbor )
    {
        /* calculate the bond vector between the neighbor and this particle
         * neighbor - particle = ( dx, dy, dz ) */
        int32_t const dx = mPolymerSystem[ 4*mNeighbors[i].bondsMonomerIdx[ iNeighbor ]+0 ] - mPolymerSystem[ 4*i+0 ];
        int32_t const dy = mPolymerSystem[ 4*mNeighbors[i].bondsMonomerIdx[ iNeighbor ]+1 ] - mPolymerSystem[ 4*i+1 ];
        int32_t const dz = mPolymerSystem[ 4*mNeighbors[i].bondsMonomerIdx[ iNeighbor ]+2 ] - mPolymerSystem[ 4*i+2 ];

        int erroneousAxis = -1;
        if ( ! ( -3 <= dx && dx <= 3 ) ) erroneousAxis = 0;
        if ( ! ( -3 <= dy && dy <= 3 ) ) erroneousAxis = 1;
        if ( ! ( -3 <= dz && dz <= 3 ) ) erroneousAxis = 2;
        if ( erroneousAxis >= 0 || mForbiddenBonds[ linearizeBondVectorIndex( dx, dy, dz ) ] )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::checkSystem] ";
            if ( erroneousAxis > 0 )
                msg << "Invalid " << 'X' + erroneousAxis << "Bond: ";
            if ( mForbiddenBonds[ linearizeBondVectorIndex( dx, dy, dz ) ] )
                msg << "This particular bond is forbidden: ";
            msg << "(" << dx << "," << dy<< "," << dz << ") between monomer "
                << i+1 << " at (" << mPolymerSystem[ 4*i+0 ] << ","
                                  << mPolymerSystem[ 4*i+1 ] << ","
                                  << mPolymerSystem[ 4*i+2 ] << ") and monomer "
                << mNeighbors[i].bondsMonomerIdx[ iNeighbor ]+1 << " at ("
                << mPolymerSystem[ 4*mNeighbors[i].bondsMonomerIdx[ iNeighbor ]+0 ] << ","
                << mPolymerSystem[ 4*mNeighbors[i].bondsMonomerIdx[ iNeighbor ]+1 ] << ","
                << mPolymerSystem[ 4*mNeighbors[i].bondsMonomerIdx[ iNeighbor ]+2 ] << ")"
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

    long int const nThreads        = 256;
    long int const nBlocksSpeciesA = ceilDiv( nMonomersSpeciesA, nThreads );
    long int const nBlocksSpeciesB = ceilDiv( nMonomersSpeciesB, nThreads );

    /* run simulation */
    for ( int32_t iStep = 1; iStep <= nMonteCarloSteps; ++iStep )
    {
        /* one Monte-Carlo step */
        for ( uint32_t iSubStep = 0; iSubStep < 2; ++iSubStep )
        {
            /* randomly choose whether to advance monomers groupt to A or B */
            switch ( randomNumbers.r250_rand32() % 2 )
            {
                case 0:
                    kernelSimulationScBFMCheckSpecies
                    <<< nBlocksSpeciesA, nThreads >>>(
                        mPolymerSystem_device, mLatticeTmp_device,
                        MonoInfo_device, mMonomerIdsA->texture,
                        nMonomersSpeciesA, randomNumbers.r250_rand32(),
                        texLatticeRefOut
                    );
                    CUDA_CHECK( cudaDeviceSynchronize() );
                    kernelSimulationScBFMPerformSpecies
                    <<< nBlocksSpeciesA, nThreads >>>(
                        mPolymerSystem_device, mLatticeOut_device,
                        mMonomerIdsA->texture, nMonomersSpeciesA,
                        texLatticeTmpRef
                    );
                    CUDA_CHECK( cudaDeviceSynchronize() );
                    kernelSimulationScBFMZeroArraySpecies
                    <<< nBlocksSpeciesA, nThreads >>>(
                        mPolymerSystem_device, mLatticeTmp_device,
                        mMonomerIdsA->texture, nMonomersSpeciesA
                    );
                    CUDA_CHECK( cudaDeviceSynchronize() );
                    break;

                case 1:
                    kernelSimulationScBFMCheckSpecies
                    <<< nBlocksSpeciesB, nThreads >>>(
                        mPolymerSystem_device, mLatticeTmp_device,
                        MonoInfo_device, mMonomerIdsB->texture,
                        nMonomersSpeciesB, randomNumbers.r250_rand32(),
                        texLatticeRefOut
                    );
                    CUDA_CHECK( cudaDeviceSynchronize() );
                    kernelSimulationScBFMPerformSpecies
                    <<< nBlocksSpeciesB, nThreads >>>(
                        mPolymerSystem_device, mLatticeOut_device,
                        mMonomerIdsB->texture, nMonomersSpeciesB,
                        texLatticeTmpRef
                    );
                    CUDA_CHECK( cudaDeviceSynchronize() );
                    kernelSimulationScBFMZeroArraySpecies
                    <<< nBlocksSpeciesB, nThreads >>>(
                        mPolymerSystem_device, mLatticeTmp_device,
                        mMonomerIdsB->texture, nMonomersSpeciesB
                    );
                    CUDA_CHECK( cudaDeviceSynchronize() );
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
    CUDA_CHECK( cudaMemcpy( mPolymerSystem, mPolymerSystem_device, ( 4*nAllMonomers+1 ) * sizeof( intCUDA ), cudaMemcpyDeviceToHost ) );

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
    /* copy information / results from GPU to Host. Actually not really
     * needed, because this is done after each kernel run i.e. the only thing
     * which should be able to touch the GPU data. (already deleted mPolymerSystem!) */
    CUDA_CHECK( cudaMemcpy( mLattice, mLatticeOut_device, mBoxX * mBoxY * mBoxZ * sizeof(uint8_t), cudaMemcpyDeviceToHost ) );

    /* check whether connectivities on GPU got corrupted */
    int sizeMonoInfo = nAllMonomers * sizeof( MonoInfo );
    CUDA_CHECK( cudaMemcpy( MonoInfo_host, MonoInfo_device, sizeMonoInfo, cudaMemcpyDeviceToHost ) );
    for ( uint32_t i = 0; i < nAllMonomers; ++i )
    {
        //if(MonoInfo_host[i].size != mNeighbors[i].size)
        if (  ( ( mPolymerSystem[ 4*i+3 ] & 224 ) >> 5 ) != mNeighbors[i].size )
        {
            std::cout << "connectivity error after simulation run" << std::endl;
            std::cout << "mono:" << i << " vs " << (i) << std::endl;
            //cout << "numElements:" << MonoInfo_host[i].size << " vs " << mNeighbors[i].size << endl;
            std::cout << "numElements:" << ((mPolymerSystem[ 4*i+3 ]&224 )>>5) << " vs " << mNeighbors[i].size << std::endl;

            throw std::runtime_error("Connectivity is corrupted! Maybe your Simulation is wrong! Exiting...\n");
        }
        for ( unsigned u = 0; u < MAX_CONNECTIVITY; ++u )
        {
            if ( MonoInfo_host[i].bondsMonomerIdx[u] != mNeighbors[i].bondsMonomerIdx[u] )
            {
                std::cout << "connectivity error after simulation run" << std::endl;
                std::cout << "mono:" << i << " vs " << (i) << std::endl;

                std::cout << "bond["<< u << "]: " << MonoInfo_host[i].bondsMonomerIdx[u] << " vs " << mNeighbors[i].bondsMonomerIdx[u] << std::endl;

                throw std::runtime_error("Connectivity is corrupted! Maybe your Simulation is wrong! Exiting...\n");
            }
        }
    }
    std::cout << "no errors in connectivity matrix after simulation run" << std::endl;

    cudaFree( mLatticeOut_device          );
    cudaFree( mLatticeTmp_device          );
    cudaFree( mPolymerSystem_device       );
    cudaFree( MonoInfo_device             );

    free( MonoInfo_host             );
    free( mLatticeOut_host          );
    free( mLatticeTmp_host          );

    if ( mPolymerSystem != NULL ){ delete[] mPolymerSystem; mPolymerSystem = NULL; }
    if ( mMonomerIdsA   != NULL ){ delete[] mMonomerIdsA  ; mMonomerIdsA   = NULL; }
    if ( mMonomerIdsB   != NULL ){ delete[] mMonomerIdsB  ; mMonomerIdsB   = NULL; }
}
