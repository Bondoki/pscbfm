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
__device__ __constant__ uint32_t dcBoxXM1   ;  // mLattice size in X-1
__device__ __constant__ uint32_t dcBoxYM1   ;  // mLattice size in Y-1
__device__ __constant__ uint32_t dcBoxZM1   ;  // mLattice size in Z-1
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
        assert( isPowerOfTwo( dcBoxXM1 + 1 ) );
        assert( isPowerOfTwo( dcBoxYM1 + 1 ) );
        assert( isPowerOfTwo( dcBoxZM1 + 1 ) );
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
     * i.e. [-127 ,128] or in this case [-4,3]
     * +4 maps to the same location as -4 but is needed or else forbidden
     * bonds couldn't be detected. Larger bonds are not possible, because
     * monomers only move by 1 per step */
    assert( -4 <= x && x <= 4 );
    assert( -4 <= y && y <= 4 );
    assert( -4 <= z && z <= 4 );
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
 * Note that the old position is not removed in order to correctly check for
 * excluded volume a second time.
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
using MonomerEdges = UpdaterGPUScBFM_AB_Type::MonomerEdges;
__global__ void kernelSimulationScBFMCheckSpecies
(
    intCUDA           * const dpPolymerSystem  ,
    uint8_t           * const dpLatticeTmp     ,
    MonomerEdges      * const dpMonoInfo       ,
    cudaTextureObject_t const texSpeciesIndices,
    uint32_t            const nMonomers        ,
    uint32_t            const rSeed            ,
    cudaTextureObject_t const texLatticeRefOut
)
{
    int const linId = blockIdx.x * blockDim.x + threadIdx.x;
    if ( linId >= nMonomers )
        return;

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
        intCUDA const nN_X = tex1Dfetch( mPolymerSystem_texture, 4*dpMonoInfo[ iMonomer ].neighborIds[ iNeighbor ]+0 );
        intCUDA const nN_Y = tex1Dfetch( mPolymerSystem_texture, 4*dpMonoInfo[ iMonomer ].neighborIds[ iNeighbor ]+1 );
        intCUDA const nN_Z = tex1Dfetch( mPolymerSystem_texture, 4*dpMonoInfo[ iMonomer ].neighborIds[ iNeighbor ]+2 );
        if ( dpForbiddenBonds[ linearizeBondVectorIndex( nN_X - x0 - dx, nN_Y - y0 - dy, nN_Z - z0 - dz ) ] )
            return;
    }

    if ( checkFront( texLatticeRefOut, x0, y0, z0, direction ) )
        return;

    /* everything fits so perform move on temporary lattice */
    /* can I do this ??? dpPolymerSystem is the device pointer to the read-only
     * texture used above. Won't this result in read-after-write race-conditions?
     * Then again the written / changed bits are never used in the above code ... */
    dpPolymerSystem[ 4*iMonomer+3 ] = properties | ( ( direction << 2 ) + 1 );
    dpLatticeTmp[ linearizeBoxVectorIndex( x0+dx, y0+dy, z0+dz ) ] = 1;
}


/**
 * Recheck whether the move is possible without collision, using the
 * temporarily parallel executed moves saved in texLatticeTmp. If so,
 * do the move in dpLattice. (Still not applied in dpPolymerSystem!)
 */
__global__ void kernelSimulationScBFMPerformSpecies
(
    intCUDA             * const dpPolymerSystem ,
    uint8_t             * const dpLattice       ,
    cudaTextureObject_t   const texSpeciesIndices,
    uint32_t              const nMonomers        ,
    cudaTextureObject_t   const texLatticeTmp
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

    if ( checkFront( texLatticeTmp, x0, y0, z0, direction ) )
        return;

    /* If possible, perform move now on normal lattice */
    dpPolymerSystem[ 4*iMonomer+3 ] = properties | 2; // indicating allowed move
    dpLattice[ linearizeBoxVectorIndex( x0+dx, y0+dy, z0+dz ) ] = 1;
    dpLattice[ linearizeBoxVectorIndex( x0   , y0   , z0    ) ] = 0;
    /* We can't clean the temporary lattice in here, because it still is being
     * used for checks. For cleaning we need only the new positions.
     * Every thread reads and writes to the same index in dpPolymerSystem.
     * From these statements follow, that we can already move the monomers
     * in this kernel, but have to watch out how we clean the temporary
     * lattice in the new kernel. For some reason it doesn't work in practice
     * though ??? ??? */
#define TEST_EARLIER_APPLY 0
#if TEST_EARLIER_APPLY > 0
    dpPolymerSystem[ 4*iMonomer+0 ] = x0 + dx;
    dpPolymerSystem[ 4*iMonomer+1 ] = y0 + dy;
    dpPolymerSystem[ 4*iMonomer+2 ] = z0 + dz;
#endif
}

/**
 * Apply move to dpPolymerSystem and clean the temporary lattice of moves
 * which seemed like they would work, but did clash with another parallel
 * move, unfortunately.
 */
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

#if TEST_EARLIER_APPLY > 0
    dpLatticeTmp[ linearizeBoxVectorIndex( x0, y0, z0 ) ] = 0;
#else
    uintCUDA const direction = ( properties & 28 ) >> 2;

    intCUDA const dx = DXTable_d[ direction ];
    intCUDA const dy = DYTable_d[ direction ];
    intCUDA const dz = DZTable_d[ direction ];

    /* possible move which clashes with another parallely moved monomer
     * Clean up the temporary lattice with these moves. */
    if ( ( properties & 3 ) == 3 )  // 3=0b11
    {
        dpPolymerSystem[ 4*iMonomer+0 ] = x0 + dx;
        dpPolymerSystem[ 4*iMonomer+1 ] = y0 + dy;
        dpPolymerSystem[ 4*iMonomer+2 ] = z0 + dz;
    }
    dpLatticeTmp[ linearizeBoxVectorIndex( x0+dx, y0+dy, z0+dz ) ] = 0;
#endif
    dpPolymerSystem[ 4*iMonomer+3 ] = properties & MASK5BITS; // delete the first 5 bits
}

UpdaterGPUScBFM_AB_Type::UpdaterGPUScBFM_AB_Type()
 : nAllMonomers         ( 0 ),
   nStars               ( 0 ),
   nMonomersPerStarArm  ( 0 ),
   nCrosslinker         ( 0 ),
   mLattice             ( NULL ),
   mLatticeOut          ( NULL ),
   mLatticeTmp          ( NULL ),
   mPolymerSystem       ( NULL ),
   mPolymerSystem_device( NULL ),
   mAttributeSystem     ( NULL ),
   mNeighbors           ( NULL ),
   mMonoInfo            ( NULL ),
   mMonomerIdsA         ( NULL ),
   mMonomerIdsB         ( NULL ),
   mBoxX                ( 0 ),
   mBoxY                ( 0 ),
   mBoxZ                ( 0 ),
   mBoxXM1              ( 0 ),
   mBoxYM1              ( 0 ),
   mBoxZM1              ( 0 ),
   mBoxXLog2            ( 0 ),
   mBoxXYLog2           ( 0 ),
   nMonomersSpeciesA    ( 0 ),
   nMonomersSpeciesB    ( 0 )
{}

/**
 * Deletes everything which could and is allocated
 */
void UpdaterGPUScBFM_AB_Type::destruct()
{
    if ( mLattice         != NULL ){ delete[] mLattice        ; mLattice         = NULL; }  // setLatticeSize
    if ( mLatticeOut      != NULL ){ delete   mLatticeOut     ; mLatticeOut      = NULL; }  // initialize
    if ( mLatticeTmp      != NULL ){ delete   mLatticeTmp     ; mLatticeTmp      = NULL; }  // initialize
    if ( mPolymerSystem   != NULL ){ delete[] mPolymerSystem  ; mPolymerSystem   = NULL; }  // setNrOfAllMonomers
    if ( mPolymerSystem_device != NULL ){ cudaFree( mPolymerSystem_device ); mPolymerSystem_device = NULL; } // initiailize
    if ( mAttributeSystem != NULL ){ delete[] mAttributeSystem; mAttributeSystem = NULL; }  // setNrOfAllMonomers
    if ( mNeighbors       != NULL ){ delete   mNeighbors      ; mNeighbors       = NULL; }  // setNrOfAllMonomers
    if ( mMonoInfo        != NULL ){ delete   mMonoInfo       ; mMonoInfo        = NULL; }  // initialize
    if ( mMonomerIdsA     != NULL ){ delete   mMonomerIdsA    ; mMonomerIdsA     = NULL; }  // initialize
    if ( mMonomerIdsB     != NULL ){ delete   mMonomerIdsB    ; mMonomerIdsB     = NULL; }  // initialize
}

UpdaterGPUScBFM_AB_Type::~UpdaterGPUScBFM_AB_Type()
{
    this->destruct();
}

void UpdaterGPUScBFM_AB_Type::initialize( int iGpuToUse )
{
    if ( mLatticeOut != NULL || mLatticeTmp != NULL || mMonoInfo != NULL ||
         mMonomerIdsA != NULL || mMonomerIdsB != NULL )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::initialize] "
            << "Initialize was already called and may not be called again "
            << "until cleanup was called!";
        throw std::runtime_error( msg.str() );
    }

    int nGpus;
    getCudaDeviceProperties( NULL, &nGpus, true /* print GPU information */ );
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

    /* count monomers per species before allocating per species arrays and
     * sorting the monomers into them */
    nMonomersSpeciesA = 0;
    nMonomersSpeciesB = 0;
    for ( uint32_t i = 0; i < nAllMonomers; ++i )
    {
        nMonomersSpeciesA += mAttributeSystem[i] == 1;
        nMonomersSpeciesB += mAttributeSystem[i] == 2;
        if ( mAttributeSystem[i] != 1 && mAttributeSystem[i] != 2 )
            throw std::runtime_error( "Wrong attribute! Only 1 or 2 allowed." );
    }

    mMonomerIdsA = new MirroredTexture< uint32_t >( nMonomersSpeciesA );
    mMonomerIdsB = new MirroredTexture< uint32_t >( nMonomersSpeciesB );

    /* sort monomers (their indices) into corresponding species array  */
    uint32_t nMonomersWrittenA = 0;
    uint32_t nMonomersWrittenB = 0;
    for ( uint32_t i = 0; i < nAllMonomers; ++i )
    {
        if ( mAttributeSystem[i] == 1 )
            mMonomerIdsA->host[ nMonomersWrittenA++ ] = i;
        else if ( mAttributeSystem[i] == 2 )
            mMonomerIdsB->host[ nMonomersWrittenB++ ] = i;
    }
    mMonomerIdsA->push();
    mMonomerIdsB->push();

    // prepare and copy the connectivity matrix to GPU
    // the index on GPU starts at 0 and is one less than loaded
    if ( mMonoInfo == NULL )
        mMonoInfo = new MirroredVector< MonomerEdges >( nAllMonomers );

    /* add property tags for each monomer with number of neighbor information */
    for ( uint32_t i = 0; i < nAllMonomers; ++i )
    {
        if ( mNeighbors->host[i].size > 7)
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::initialize] "
                << "This implementation allows max. 7 neighbors per monomer, "
                << "but monomer " << i << " has " << mNeighbors->host[i].size << "\n";
            throw std::invalid_argument( msg.str() );
        }
        mPolymerSystem[ 4*i+3 ] |= ( (intCUDA) mNeighbors->host[i].size ) << 5;
    }
    if ( mNeighbors == NULL )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::initialize] "
            << "mNeighbors is still NULL! setNrOfAllMonomers needs to be called first!\n";
        throw std::invalid_argument( msg.str() );
    }
    std::memcpy( mMonoInfo->host, mNeighbors->host, mMonoInfo->nBytes );
    mMonoInfo->push();
    mNeighbors->push();
    std::cerr << "mMonoInfo = " << *mMonoInfo << "\n";
    std::cerr << "mNeighbors = " << *mNeighbors << "\n";

    checkSystem();

    /* creating lattice */
    CUDA_CHECK( cudaMemcpyToSymbol( dcBoxXM1   , &mBoxXM1   , sizeof( mBoxXM1    ) ) );
    CUDA_CHECK( cudaMemcpyToSymbol( dcBoxYM1   , &mBoxYM1   , sizeof( mBoxYM1    ) ) );
    CUDA_CHECK( cudaMemcpyToSymbol( dcBoxZM1   , &mBoxZM1   , sizeof( mBoxZM1    ) ) );
    CUDA_CHECK( cudaMemcpyToSymbol( dcBoxXLog2 , &mBoxXLog2 , sizeof( mBoxXLog2  ) ) );
    CUDA_CHECK( cudaMemcpyToSymbol( dcBoxXYLog2, &mBoxXYLog2, sizeof( mBoxXYLog2 ) ) );

    mLatticeOut = new MirroredTexture< uint8_t >( mBoxX * mBoxY * mBoxZ );
    mLatticeTmp = new MirroredTexture< uint8_t >( mBoxX * mBoxY * mBoxZ );
    CUDA_CHECK( cudaMemset( mLatticeTmp->gpu, 0, mLatticeTmp->nBytes ) );
    /* populate latticeOut with monomers from mPolymerSystem */
    std::memset( mLatticeOut->host, 0, mLatticeOut->nBytes );
    for ( uint32_t t = 0; t < nAllMonomers; ++t )
    {
        #ifdef USEZCURVE
            uint32_t xk = mPolymerSystem[ 4*t+0 ] & mBoxXM1;
            uint32_t yk = mPolymerSystem[ 4*t+1 ] & mBoxYM1;
            uint32_t zk = mPolymerSystem[ 4*t+2 ] & mBoxZM1;
            uint32_t inter3 = interleave3( xk/2 , yk/2, zk/2 );
            mLatticeOut_host[ ( ( mPolymerSystem_host[ 4*t+3 ] & 1 ) << 23 ) + inter3 ] = 1;
        #else
        mLatticeOut->host[ linearizeBoxVectorIndex( mPolymerSystem[ 4*t+0 ],
                                                    mPolymerSystem[ 4*t+1 ],
                                                    mPolymerSystem[ 4*t+2 ] ) ] = 1;
        #endif
    }
    mLatticeOut->push();
    CUDA_CHECK( cudaMalloc( (void **) &mPolymerSystem_device, ( 4*nAllMonomers+1 ) * sizeof( intCUDA ) ) );
    CUDA_CHECK( cudaMemcpy( mPolymerSystem_device, mPolymerSystem, ( 4*nAllMonomers+1 ) * sizeof( intCUDA ), cudaMemcpyHostToDevice ) );
    cudaBindTexture( 0, mPolymerSystem_texture, mPolymerSystem_device, ( 4*nAllMonomers+1 ) * sizeof( intCUDA ) );
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
    mAttributeSystem = new int32_t[ nAllMonomers   ];
    mPolymerSystem   = new intCUDA[ nAllMonomers*4 ];
    mNeighbors       = new MirroredVector< MonomerEdges >( nAllMonomers );
    std::memset( mNeighbors->host, 0, mNeighbors->nBytes );
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
    /* @todo add check whether the bond already exists */
    /* Could also add the inversio, but the bonds are a non-directional graph */
    auto const iNew = mNeighbors->host[ iMonomer1 ].size++;
    if ( iNew > MAX_CONNECTIVITY-1 )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setConnectivity" << "] "
            << "The maximum amount of bonds per monomer (" << MAX_CONNECTIVITY
            << ") has been exceeded!\n";
        throw std::invalid_argument( msg.str() );
    }
    mNeighbors->host[ iMonomer1 ].neighborIds[ iNew ] = iMonomer2;
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
    if ( mBoxX != ( 1u << mBoxXLog2 ) || mBoxX * boxY != ( 1u << mBoxXYLog2 ) )
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
    for ( uint32_t i = 0; i < nAllMonomers; ++i )
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
    for ( unsigned iNeighbor = 0; iNeighbor < mNeighbors->host[i].size; ++iNeighbor )
    {
        /* calculate the bond vector between the neighbor and this particle
         * neighbor - particle = ( dx, dy, dz ) */
        intCUDA * const neighbor = & mPolymerSystem[ 4*mNeighbors->host[i].neighborIds[ iNeighbor ] ];
        int32_t const dx = neighbor[0] - mPolymerSystem[ 4*i+0 ];
        int32_t const dy = neighbor[1] - mPolymerSystem[ 4*i+1 ];
        int32_t const dz = neighbor[2] - mPolymerSystem[ 4*i+2 ];

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
                << mNeighbors->host[i].neighborIds[ iNeighbor ]+1 << " at ("
                << neighbor[0] << "," << neighbor[1] << "," << neighbor[2] << ")"
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
            int nBlocks = 0;
            uint32_t nMonomersSpecies = 0;
            MirroredTexture< uint32_t > * monomerIds = NULL;
            if ( randomNumbers.r250_rand32() % 2 == 0 )
            {
                nBlocks          = nBlocksSpeciesA  ;
                monomerIds       = mMonomerIdsA     ;
                nMonomersSpecies = nMonomersSpeciesA;
            }
            else
            {
                nBlocks          = nBlocksSpeciesB  ;
                monomerIds       = mMonomerIdsB     ;
                nMonomersSpecies = nMonomersSpeciesB;
            }
            /* WHY CAN'T I REPLACE mMonoInfo->gpu with mNeighbors->gpu ??? */
            kernelSimulationScBFMCheckSpecies
            <<< nBlocks, nThreads >>>(
                mPolymerSystem_device, mLatticeTmp->gpu,
                mMonoInfo->gpu, monomerIds->texture,
                nMonomersSpecies, randomNumbers.r250_rand32(),
                mLatticeOut->texture
            );
            CUDA_CHECK( cudaDeviceSynchronize() );  // for debug purposes. Might hinder performance
            kernelSimulationScBFMPerformSpecies
            <<< nBlocks, nThreads >>>(
                mPolymerSystem_device, mLatticeOut->gpu,
                monomerIds->texture, nMonomersSpecies,
                mLatticeTmp->texture
            );
            CUDA_CHECK( cudaDeviceSynchronize() );
            kernelSimulationScBFMZeroArraySpecies
            <<< nBlocks, nThreads >>>(
                mPolymerSystem_device, mLatticeTmp->gpu,
                monomerIds->texture, nMonomersSpecies
            );
            CUDA_CHECK( cudaDeviceSynchronize() );
        }
    }

    /* all MCS are done- copy information back from GPU to host */
    mLatticeTmp->pop();
    unsigned nOccupied = 0;
    for ( unsigned i = 0u; i < mBoxX * mBoxY * mBoxZ; ++i )
        nOccupied += mLatticeTmp->host[i] != 0;
    if ( nOccupied != 0 )
    {
        std::stringstream msg;
        msg << "latticeTmp occupation (" << nOccupied << ") should be 0! Exiting ...\n";
        throw std::runtime_error( msg.str() );
    }

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

/**
 * GPUScBFM_AB_Type::initialize and run and cleanup should be usable on
 * repeat. Which means we need to destruct everything created in
 * GPUScBFM_AB_Type::initialize, which encompasses setLatticeSize,
 * setNrOfAllMonomers and initialize. Currently this includes all allocs,
 * so we can simply call destruct.
 */
void UpdaterGPUScBFM_AB_Type::cleanup()
{
    /* check whether connectivities on GPU got corrupted */
    for ( uint32_t i = 0; i < nAllMonomers; ++i )
    {
        unsigned const nNeighbors = ( mPolymerSystem[ 4*i+3 ] & 224 /* 0b11100000 */ ) >> 5;
        if ( nNeighbors != mNeighbors->host[i].size )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::~cleanup" << "] "
                << "Connectivities in property field of mPolymerSystem are "
                << "different from host-side connectivities. This should not "
                << "happen! (Monomer " << i << ": " << nNeighbors << " != "
                << mNeighbors->host[i].size << "\n";
            throw std::runtime_error( msg.str() );
        }
    }
    this->destruct();
}
