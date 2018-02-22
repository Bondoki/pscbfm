/*
 * UpdaterGPUScBFM_AB_Type.cpp
 *
 *  Created on: 27.07.2017
 *      Author: Ron Dockhorn
 */

#include "UpdaterGPUScBFM_AB_Type.h"


//#define USE_THRUST_FILL
#define USE_BIT_PACKING_TMP_LATTICE
//#define USE_BIT_PACKING_LATTICE
//#define AUTO_CONFIGURE_BEST_SETTINGS_FOR_PSCBFM_ALGORITHM


#include <algorithm>                        // fill, sort
#include <chrono>                           // std::chrono::high_resolution_clock
#include <cstdio>                           // printf
#include <cstdlib>                          // exit
#include <cstring>                          // memset
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <stdint.h>
#include <sstream>

#include <cuda_profiler_api.h>              // cudaProfilerStop
#ifdef USE_THRUST_FILL
#   include <thrust/system/cuda/execution_policy.h>
#   include <thrust/fill.h>
#endif

#include "cudacommon.hpp"
#include "SelectiveLogger.hpp"
#include "graphColoring.tpp"

#define DEBUG_UPDATERGPUSCBFM_AB_TYPE 100
#if defined( USE_BIT_PACKING_TMP_LATTICE ) || defined( USE_BIT_PACKING_LATTICE )
#   define USE_BIT_PACKING
#endif

/* 512=8^3 for a range of bonds per direction of [-4,3] */
__device__ __constant__ bool dpForbiddenBonds[512]; //false-allowed; true-forbidden

/**
 * These will be initialized to:
 *   DXTable_d = { -1,1,0,0,0,0 }
 *   DYTable_d = { 0,0,-1,1,0,0 }
 *   DZTable_d = { 0,0,0,0,-1,1 }
 * I.e. a table of three random directional 3D vectors \vec{dr} = (dx,dy,dz)
 */
__device__ __constant__ uint32_t DXTable_d[6]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
__device__ __constant__ uint32_t DYTable_d[6]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
__device__ __constant__ uint32_t DZTable_d[6]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
/**
 * If intCUDA is different from uint32_t, then this second table prevents
 * expensive type conversions, but both tables are still needed, the
 * uint32_t version, because the calculation of the linear index will result
 * in uint32_t anyway and the intCUDA version for solely updating the
 * position information
 */
__device__ __constant__ intCUDA DXTableIntCUDA_d[6];
__device__ __constant__ intCUDA DYTableIntCUDA_d[6];
__device__ __constant__ intCUDA DZTableIntCUDA_d[6];

/* will this really bring performance improvement? At least constant cache
 * might be as fast as register access when all threads in a warp access the
 * the same constant */
__device__ __constant__ uint32_t dcBoxXM1   ;  // mLattice size in X-1
__device__ __constant__ uint32_t dcBoxYM1   ;  // mLattice size in Y-1
__device__ __constant__ uint32_t dcBoxZM1   ;  // mLattice size in Z-1
__device__ __constant__ uint32_t dcBoxXLog2 ;  // mLattice shift in X
__device__ __constant__ uint32_t dcBoxXYLog2;  // mLattice shift in X*Y


/* for a in-depth description see comments in Fundamental/BitsCompileTime.hpp
 * where it was copied from */
template< typename T, unsigned char nSpacing, unsigned char nStepsNeeded, unsigned char iStep >
struct DiluteBitsCrumble { __device__ __host__ inline static T apply( T const & xLastStep )
{
    auto x = DiluteBitsCrumble<T,nSpacing,nStepsNeeded,iStep-1>::apply( xLastStep );
    auto constexpr iStep2Pow = 1llu << ( (nStepsNeeded-1) - iStep );
    auto constexpr mask = BitPatterns::RectangularWave< T, iStep2Pow, iStep2Pow * nSpacing >::value;
    x = ( x + ( x << ( iStep2Pow * nSpacing ) ) ) & mask;
    return x;
} };

template< typename T, unsigned char nSpacing, unsigned char nStepsNeeded >
struct DiluteBitsCrumble<T,nSpacing,nStepsNeeded,0> { __device__ __host__ inline static T apply( T const & x )
{
    auto constexpr nBitsAllowed = 1 + ( sizeof(T) * CHAR_BIT - 1 ) / ( nSpacing + 1 );
    return x & BitPatterns::Ones< T, nBitsAllowed >::value;
} };

template< typename T, unsigned char nSpacing >
__device__ __host__ inline T diluteBits( T const & rx )
{
    static_assert( nSpacing > 0, "" );
    auto constexpr nBitsAvailable = sizeof(T) * CHAR_BIT;
    static_assert( nBitsAvailable > 0, "" );
    auto constexpr nBitsAllowed = CompileTimeFunctions::ceilDiv( nBitsAvailable, nSpacing + 1 );
    auto constexpr nStepsNeeded = 1 + CompileTimeFunctions::CeilLog< 2, nBitsAllowed >::value;
    return DiluteBitsCrumble< T, nSpacing, nStepsNeeded, ( nStepsNeeded > 0 ? nStepsNeeded-1 : 0 ) >::apply( rx );
}

/**
 * Legacy function which ironically might be more readable than my version
 * which derives and thereby documents in-code where the magic constants
 * derive from :(
 * Might be needed to compare performance to the template version.
 *  => is slower by 1%
 * Why is it using ^ instead of | ??? !!!
 */
/*
__device__ uint32_t part1by2_d( uint32_t n )
{
    n&= 0x000003ff;
    n = (n ^ (n << 16)) & 0xff0000ff; // 0b 0000 0000 1111 1111
    n = (n ^ (n <<  8)) & 0x0300f00f; // 0b 1111 0000 0000 1111
    n = (n ^ (n <<  4)) & 0x030c30c3; // 0b 0011 0000 1100 0011
    n = (n ^ (n <<  2)) & 0x09249249; // 0b 1001 0010 0100 1001
    return n;
}
*/

namespace {

template< typename T >
__device__ __host__ bool isPowerOfTwo( T const & x )
{
    //return ! ( x == T(0) ) && ! ( x & ( x - T(1) ) );
    return __popc( x ) <= 1;
}

}

#define USE_ZCURVE_FOR_LATTICE
uint32_t UpdaterGPUScBFM_AB_Type::linearizeBoxVectorIndex
(
    uint32_t const & ix,
    uint32_t const & iy,
    uint32_t const & iz
)
{
    #if defined ( USE_ZCURVE_FOR_LATTICE )
        return   diluteBits< uint32_t, 2 >( ix & mBoxXM1 )        +
               ( diluteBits< uint32_t, 2 >( iy & mBoxYM1 ) << 1 ) +
               ( diluteBits< uint32_t, 2 >( iz & mBoxZM1 ) << 2 );
    #elif defined( NOMAGIC )
        return ( ix % mBoxX ) +
               ( iy % mBoxY ) * mBoxX +
               ( iz % mBoxZ ) * mBoxX * mBoxY;
    #else
        #if DEBUG_UPDATERGPUSCBFM_AB_TYPE > 10
            assert( isPowerOfTwo( mBoxXM1 + 1 ) );
            assert( isPowerOfTwo( mBoxYM1 + 1 ) );
            assert( isPowerOfTwo( mBoxZM1 + 1 ) );
        #endif
        return   ( ix & mBoxXM1 ) +
               ( ( iy & mBoxYM1 ) << mBoxXLog2  ) +
               ( ( iz & mBoxZM1 ) << mBoxXYLog2 );
    #endif
}


#define USE_BIT_PACKING
#ifdef USE_BIT_PACKING
    template< typename T > __device__ __host__ inline
    T bitPackedGet( T const * const & p, uint32_t const & i )
    {
        /**
         * >> 3, because 3 bits = 2^3=8 numbers are used for sub-byte indexing,
         * i.e. we divide the index i by 8 which is equal to the space we save
         * by bitpacking.
         * & 7, because 7 = 0b111, i.e. we are only interested in the last 3
         * bits specifying which subbyte element we want
         */
        return ( p[ i >> 3 ] >> ( i & T(7) ) ) & T(1);
    }

    template< typename T > __device__ inline
    T bitPackedTextureGet( cudaTextureObject_t const & p, uint32_t const & i )
    {
        return ( tex1Dfetch<T>( p, i >> 3 ) >> ( i & T(7) ) ) & T(1);
    }

    /**
     * Because the smalles atomic is for int (4x uint8_t) we need to
     * cast the array to that and then do a bitpacking for the whole 32 bits
     * instead of 8 bits
     * I.e. we need to address 32 subbits, i.e. >>3 becomes >>5
     * and &7 becomes &31 = 0b11111 = 0x1F
     * __host__ __device__ function with differing code
     * @see https://codeyarns.com/2011/03/14/cuda-common-function-for-both-host-and-device-code/
     */
    template< typename T > __device__ __host__ inline
    void bitPackedSet( T * const __restrict__ p, uint32_t const & i )
    {
        static_assert( sizeof(int) == 4, "" );
        #ifdef __CUDA_ARCH__
            atomicOr ( (int*) p + ( i >> 5 ),    T(1) << ( i & T( 0x1F ) )   );
        #else
            p[ i >> 3 ] |= T(1) << ( i & T(7) );
        #endif
    }

    template< typename T > __device__ __host__ inline
    void bitPackedUnset( T * const __restrict__ p, uint32_t const & i )
    {
        #ifdef __CUDA_ARCH__
            atomicAnd( (int*) p + ( i >> 5 ), ~( T(1) << ( i & T( 0x1F ) ) ) );
        #else
            p[ i >> 3 ] &= ~( T(1) << ( i & T(7) ) );
        #endif
    }
#else
    template< typename T > __device__ __host__ inline
    T bitPackedGet( T const * const & p, uint32_t const & i ){ return p[i]; }
    template< typename T > __device__ inline
    T bitPackedTextureGet( cudaTextureObject_t const & p, uint32_t const & i ) {
        return tex1Dfetch<T>(p,i); }
    template< typename T > __device__ __host__ inline
    void bitPackedSet  ( T * const __restrict__ p, uint32_t const & i ){ p[i] = 1; }
    template< typename T > __device__ __host__ inline
    void bitPackedUnset( T * const __restrict__ p, uint32_t const & i ){ p[i] = 0; }
#endif


using T_Flags = UpdaterGPUScBFM_AB_Type::T_Flags;

__device__ inline bool checkFrontBitPacked
(
    cudaTextureObject_t const & texLattice,
    uint32_t            const & x0        ,
    uint32_t            const & y0        ,
    uint32_t            const & z0        ,
    intCUDA             const & axis
);

__device__ inline uint32_t linearizeBoxVectorIndex
(
    uint32_t const & ix,
    uint32_t const & iy,
    uint32_t const & iz
);


/**
 * Recheck whether the move is possible without collision, using the
 * temporarily parallel executed moves saved in texLatticeTmp. If so,
 * do the move in dpLattice. (Still not applied in dpPolymerSystem!)
 */
__global__ void kernelSimulationScBFMPerformSpecies
(
    vecIntCUDA    const * const __restrict__ dpPolymerSystem,
    T_Flags             * const __restrict__ dpPolymerFlags ,
    uint8_t             * const __restrict__ dpLattice      ,
    uint32_t              const              nMonomers      ,
    cudaTextureObject_t   const              texLatticeTmp
)
{
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x )
    {
        auto const properties = dpPolymerFlags[ iMonomer ];
        if ( ( properties & T_Flags(1) ) == T_Flags(0) )    // impossible move
            continue;

        auto const r0 = dpPolymerSystem[ iMonomer ];
        //uint3 const r0 = { r0Raw.x, r0Raw.y, r0Raw.z }; // slower
        auto const direction = ( properties >> T_Flags(2) ) & T_Flags(7); // 7=0b111
    #ifdef USE_BIT_PACKING_TMP_LATTICE
        if ( checkFrontBitPacked( texLatticeTmp, r0.x, r0.y, r0.z, direction ) )
    #else
        if ( checkFront( texLatticeTmp, r0.x, r0.y, r0.z, direction ) )
    #endif
            continue;

        /* If possible, perform move now on normal lattice */
        dpPolymerFlags[ iMonomer ] = properties | T_Flags(2); // indicating allowed move
        dpLattice[ linearizeBoxVectorIndex( r0.x, r0.y, r0.z ) ] = 0;
        dpLattice[ linearizeBoxVectorIndex( r0.x + DXTable_d[ direction ],
                                            r0.y + DYTable_d[ direction ],
                                            r0.z + DZTable_d[ direction ] ) ] = 1;
        /* We can't clean the temporary lattice in here, because it still is being
         * used for checks. For cleaning we need only the new positions.
         * But we can't use the applied positions, because we also need to clean
         * those particles which couldn't move in this second kernel but where
         * still set in the lattice by the first kernel! */
    }
}

/**
 * Apply move to dpPolymerSystem and clean the temporary lattice of moves
 * which seemed like they would work, but did clash with another parallel
 * move, unfortunately.
 * @todo it might be better to just use a cudaMemset to clean the lattice,
 *       that way there wouldn't be any memory dependencies and calculations
 *       needed, even though we would have to clean everything, instead of
 *       just those set. But that doesn't matter, because most of the threads
 *       are idling anyway ...
 *       This kind of kernel might give some speedup after stream compaction
 *       has been implemented though.
 *    -> print out how many percent of cells need to be cleaned .. I need
 *       many more statistics anyway for evaluating performance benefits a bit
 *       better!
 */
__global__ void kernelSimulationScBFMZeroArraySpecies
(
    vecIntCUDA          * const __restrict__ dpPolymerSystem,
    T_Flags       const * const __restrict__ dpPolymerFlags ,
    uint8_t             * const __restrict__ dpLatticeTmp   ,
    uint32_t              const              nMonomers
)
{
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x )
    {
        auto const properties = dpPolymerFlags[ iMonomer ];
        if ( ( properties & T_Flags(3) ) == T_Flags(0) )    // impossible move
            continue;

        auto r0 = dpPolymerSystem[ iMonomer ];
        auto const direction = ( properties >> T_Flags(2) ) & T_Flags(7); // 7=0b111

        r0.x += DXTableIntCUDA_d[ direction ];
        r0.y += DYTableIntCUDA_d[ direction ];
        r0.z += DZTableIntCUDA_d[ direction ];
    #ifdef USE_BIT_PACKING_TMP_LATTICE
        //bitPackedUnset( dpLatticeTmp, linearizeBoxVectorIndex( r0.x, r0.y, r0.z ) );
        dpLatticeTmp[ linearizeBoxVectorIndex( r0.x, r0.y, r0.z ) >> 3 ] = 0;
    #else
        dpLatticeTmp[ linearizeBoxVectorIndex( r0.x, r0.y, r0.z ) ] = 0;
    #endif
        if ( ( properties & T_Flags(3) ) == T_Flags(3) )  // 3=0b11
            dpPolymerSystem[ iMonomer ] = r0;
    }
}


#include "CRITICAL.cu"


UpdaterGPUScBFM_AB_Type::UpdaterGPUScBFM_AB_Type()
 : mStream              ( 0 ),
   nAllMonomers         ( 0 ),
   mLattice             ( NULL ),
   mLatticeOut          ( NULL ),
   mLatticeTmp          ( NULL ),
   mPolymerSystemSorted ( NULL ),
   mPolymerFlags        ( NULL ),
   mNeighborsSorted     ( NULL ),
   mNeighborsSortedSizes( NULL ),
   mNeighborsSortedInfo ( nBytesAlignment ),
   mAttributeSystem     ( NULL ),
   mBoxX                ( 0 ),
   mBoxY                ( 0 ),
   mBoxZ                ( 0 ),
   mBoxXM1              ( 0 ),
   mBoxYM1              ( 0 ),
   mBoxZM1              ( 0 ),
   mBoxXLog2            ( 0 ),
   mBoxXYLog2           ( 0 )
{
    /**
     * Log control.
     * Note that "Check" controls not the output, but the actualy checks
     * If a checks needs to always be done, then do that check and declare
     * the output as "Info" log level
     */
    mLog.file( __FILENAME__ );
    mLog.  activate( "Benchmark" );
    mLog.deactivate( "Check"     );
    mLog.  activate( "Error"     );
    mLog.  activate( "Info"      );
    mLog.deactivate( "Stats"     );
    mLog.deactivate( "Warning"   );
}

/**
 * Deletes everything which could and is allocated
 */
void UpdaterGPUScBFM_AB_Type::destruct()
{
    if ( mLattice         != NULL ){ delete[] mLattice        ; mLattice         = NULL; }  // setLatticeSize
    if ( mLatticeOut      != NULL ){ delete   mLatticeOut     ; mLatticeOut      = NULL; }  // initialize
    if ( mLatticeTmp      != NULL ){ delete   mLatticeTmp     ; mLatticeTmp      = NULL; }  // initialize
    if ( mPolymerSystemSorted != NULL ){ delete mPolymerSystemSorted; mPolymerSystemSorted = NULL; }  // initialize
    if ( mPolymerFlags    != NULL ){ delete   mPolymerFlags   ; mPolymerFlags    = NULL; }  // initialize
    if ( mNeighborsSorted != NULL ){ delete   mNeighborsSorted; mNeighborsSorted = NULL; }  // initialize
    if ( mNeighborsSortedSizes != NULL ){ delete   mNeighborsSortedSizes; mNeighborsSortedSizes = NULL; }  // initialize
    if ( mAttributeSystem != NULL ){ delete[] mAttributeSystem; mAttributeSystem = NULL; }  // setNrOfAllMonomers
}

UpdaterGPUScBFM_AB_Type::~UpdaterGPUScBFM_AB_Type()
{
    this->destruct();
}

void UpdaterGPUScBFM_AB_Type::setGpu( int iGpuToUse )
{
    int nGpus;
    getCudaDeviceProperties( NULL, &nGpus, true /* print GPU information */ );
    if ( ! ( 0 <= iGpuToUse && iGpuToUse < nGpus ) )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setGpu] "
            << "GPU with ID " << iGpuToUse << " not present. "
            << "Only " << nGpus << " GPUs are available.\n";
        mLog( "Error" ) << msg.str();
        throw std::invalid_argument( msg.str() );
    }
    CUDA_ERROR( cudaSetDevice( iGpuToUse ) );
    miGpuToUse = iGpuToUse;
}


void UpdaterGPUScBFM_AB_Type::initialize( void )
{
    if ( mLog( "Stats" ).isActive() )
    {
        // this is called in parallel it seems, therefore need to buffer it
        std::stringstream msg; msg
        << "[" << __FILENAME__ << "::initialize] The "
        << "(" << mBoxX << "," << mBoxY << "," << mBoxZ << ")"
        << " lattice is populated by " << nAllMonomers
        << " resulting in a filling rate of "
        << nAllMonomers / double( mBoxX * mBoxY * mBoxZ ) << "\n";
        mLog( "Stats" ) << msg.str();
    }

    if ( mLatticeOut != NULL || mLatticeTmp != NULL )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::initialize] "
            << "Initialize was already called and may not be called again "
            << "until cleanup was called!";
        mLog( "Error" ) << msg.str();
        throw std::runtime_error( msg.str() );
    }

    /* create the BondTable and copy it to constant memory */
    bool * tmpForbiddenBonds = (bool*) malloc( sizeof( bool ) * 512 );
    unsigned nAllowedBonds = 0;
    for ( int i = 0; i < 512; ++i )
        if ( ! ( tmpForbiddenBonds[i] = mForbiddenBonds[i] ) )
            ++nAllowedBonds;
    /* Why does it matter? Shouldn't it work with arbitrary bond sets ??? */
    if ( nAllowedBonds != 108 )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::initialize] "
            << "Wrong bond-set! Expected 108 allowed bonds, but got " << nAllowedBonds << "\n";
        mLog( "Error" ) << msg.str();
        throw std::runtime_error( msg.str() );
    }
    CUDA_ERROR( cudaMemcpyToSymbol( dpForbiddenBonds, tmpForbiddenBonds, sizeof(bool)*512 ) );
    free( tmpForbiddenBonds );

    /* create a table mapping the random int to directions whereto move the
     * monomers. We can use negative numbers, because (0u-1u)+1u still is 0u */
    uint32_t tmp_DXTable[6] = { 0u-1u,1,  0,0,  0,0 };
    uint32_t tmp_DYTable[6] = {  0,0, 0u-1u,1,  0,0 };
    uint32_t tmp_DZTable[6] = {  0,0,  0,0, 0u-1u,1 };
    CUDA_ERROR( cudaMemcpyToSymbol( DXTable_d, tmp_DXTable, sizeof( tmp_DXTable ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( DYTable_d, tmp_DYTable, sizeof( tmp_DXTable ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( DZTable_d, tmp_DZTable, sizeof( tmp_DXTable ) ) );
    intCUDA tmp_DXTableIntCUDA[6] = { -1,1,  0,0,  0,0 };
    intCUDA tmp_DYTableIntCUDA[6] = {  0,0, -1,1,  0,0 };
    intCUDA tmp_DZTableIntCUDA[6] = {  0,0,  0,0, -1,1 };
    CUDA_ERROR( cudaMemcpyToSymbol( DXTableIntCUDA_d, tmp_DXTableIntCUDA, sizeof( tmp_DZTableIntCUDA ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( DYTableIntCUDA_d, tmp_DYTableIntCUDA, sizeof( tmp_DZTableIntCUDA ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( DZTableIntCUDA_d, tmp_DZTableIntCUDA, sizeof( tmp_DZTableIntCUDA ) ) );

    /*************************** start of grouping ***************************/

   mLog( "Info" ) << "Coloring graph ...\n";
    bool const bUniformColors = true; // setting this to true should yield more performance as the kernels are uniformly utilized
    mGroupIds = graphColoring< MonomerEdges const *, uint32_t, uint8_t >(
        &mNeighbors[0], mNeighbors.size(), bUniformColors,
        []( MonomerEdges const * const & x, uint32_t const & i ){ return x[i].size; },
        []( MonomerEdges const * const & x, uint32_t const & i, size_t const & j ){ return x[i].neighborIds[j]; }
    );

    /* count monomers per species before allocating per species arrays and
     * sorting the monomers into them */
    mnElementsInGroup.resize(0);
    for ( size_t i = 0u; i < mGroupIds.size(); ++i )
    {
        if ( mGroupIds[i] >= mnElementsInGroup.size() )
            mnElementsInGroup.resize( mGroupIds[i]+1, 0 );
        ++mnElementsInGroup[ mGroupIds[i] ];
    }
    /**
     * Generate new array which contains all sorted monomers aligned
     * @verbatim
     * ABABABABABA
     * A A A A A A
     *  B B B B B
     * AAAAAA  BBBBB
     *        ^ alignment
     * @endverbatim
     * in the worst case we are only one element ( 4*intCUDA ) over the
     * alignment with each group and need to fill up to nBytesAlignment for
     * all of them */
    /* virtual number of monomers which includes the additional alignment padding */
    auto const nMonomersPadded = nAllMonomers + ( nElementsAlignment - 1u ) * mnElementsInGroup.size();
    assert( mPolymerFlags == NULL );
    mPolymerFlags = new MirroredVector< T_Flags >( nMonomersPadded, mStream );
    CUDA_ERROR( cudaMemset( mPolymerFlags->gpu, 0, mPolymerFlags->nBytes ) );
    /* Calculate offsets / prefix sum including the alignment */
    assert( mPolymerSystemSorted == NULL );
    mPolymerSystemSorted = new MirroredVector< vecIntCUDA >( nMonomersPadded, mStream );
    #ifndef NDEBUG
        std::memset( mPolymerSystemSorted->host, 0, mPolymerSystemSorted->nBytes );
    #endif

    /* calculate offsets to each aligned subgroup vector */
    iSubGroupOffset.resize( mnElementsInGroup.size() );
    iSubGroupOffset.at(0) = 0;
    for ( size_t i = 1u; i < mnElementsInGroup.size(); ++i )
    {
        iSubGroupOffset[i] = iSubGroupOffset[i-1] +
        ceilDiv( mnElementsInGroup[i-1], nElementsAlignment ) * nElementsAlignment;
        assert( iSubGroupOffset[i] - iSubGroupOffset[i-1] >= mnElementsInGroup[i-1] );
    }

    /* virtually sort groups into new array and save index mappings */
    iToiNew.resize( nAllMonomers   , UINT32_MAX );
    iNewToi.resize( nMonomersPadded, UINT32_MAX );
    std::vector< size_t > iSubGroup = iSubGroupOffset;   /* stores the next free index for each subgroup */
    for ( size_t i = 0u; i < nAllMonomers; ++i )
    {
        iToiNew[i] = iSubGroup[ mGroupIds[i] ]++;
        iNewToi[ iToiNew[i] ] = i;
    }

    /* adjust neighbor IDs to new sorted PolymerSystem and also sort that array.
     * Bonds are not supposed to change, therefore we don't need to push and
     * pop them each time we do something on the GPU! */
    assert( mNeighborsSorted == NULL );
    assert( mNeighborsSortedInfo.getRequiredBytes() == 0 );
    for ( size_t i = 0u; i < mnElementsInGroup.size(); ++i )
        mNeighborsSortedInfo.newMatrix( MAX_CONNECTIVITY, mnElementsInGroup[i] );
    mNeighborsSorted = new MirroredVector< uint32_t >( mNeighborsSortedInfo.getRequiredElements(), mStream );
    std::memset( mNeighborsSorted->host, 0, mNeighborsSorted->nBytes );
    mNeighborsSortedSizes = new MirroredVector< uint8_t >( nMonomersPadded, mStream );
    std::memset( mNeighborsSortedSizes->host, 0, mNeighborsSortedSizes->nBytes );

    {
        size_t iSpecies = 0u;
        /* iterate over sorted instead of unsorted array so that calculating
         * the current species we are working on is easier */
        for ( size_t i = 0u; i < iNewToi.size(); ++i )
        {
            /* check if we are already working on a new species */
            if ( iSpecies+1 < iSubGroupOffset.size() &&
                 i >= iSubGroupOffset[ iSpecies+1 ] )
            {
                ++iSpecies;
            }
            /* skip over padded indices */
            if ( iNewToi[i] >= nAllMonomers )
                continue;
            /* actually to the sorting / copying and conversion */
            mNeighborsSortedSizes->host[i] = mNeighbors[ iNewToi[i] ].size;
            auto const pitch = mNeighborsSortedInfo.getMatrixPitchElements( iSpecies );
            for ( size_t j = 0u; j < mNeighbors[  iNewToi[i] ].size; ++j )
            {
                mNeighborsSorted->host[ mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ) + j * pitch + ( i - iSubGroupOffset[ iSpecies ] ) ] = iToiNew[ mNeighbors[ iNewToi[i] ].neighborIds[j] ];
                //mNeighborsSorted->host[ iToiNew[i] ].neighborIds[j] = iToiNew[ mNeighbors[i].neighborIds[j] ];
            }
        }
    }
    mNeighborsSorted->pushAsync();
    mNeighborsSortedSizes->pushAsync();

    /************************** end of group sorting **************************/

    /* sort groups into new array and save index mappings */
    mLog( "Info" ) << "[UpdaterGPUScBFM_AB_Type::runSimulationOnGPU] sort mPolymerSystem -> mPolymerSystemSorted ... ";
    for ( size_t i = 0u; i < nAllMonomers; ++i )
    {
        if ( i < 20 )
            mLog( "Info" ) << "Write " << i << " to " << this->iToiNew[i] << "\n";
        auto const pTarget = mPolymerSystemSorted->host + iToiNew[i];
        pTarget->x = mPolymerSystem[ 4*i+0 ];
        pTarget->y = mPolymerSystem[ 4*i+1 ];
        pTarget->z = mPolymerSystem[ 4*i+2 ];
        pTarget->w = mNeighbors[i].size;
    }
    mPolymerSystemSorted->pushAsync();

    checkSystem();

    /* creating lattice */
    CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXM1   , &mBoxXM1   , sizeof( mBoxXM1    ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( dcBoxYM1   , &mBoxYM1   , sizeof( mBoxYM1    ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZM1   , &mBoxZM1   , sizeof( mBoxZM1    ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXLog2 , &mBoxXLog2 , sizeof( mBoxXLog2  ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXYLog2, &mBoxXYLog2, sizeof( mBoxXYLog2 ) ) );

    mLatticeOut = new MirroredTexture< uint8_t >( mBoxX * mBoxY * mBoxZ, mStream );
    mLatticeTmp = new MirroredTexture< uint8_t >( mBoxX * mBoxY * mBoxZ, mStream );
    CUDA_ERROR( cudaMemsetAsync( mLatticeTmp->gpu, 0, mLatticeTmp->nBytes, mStream ) );
    /* populate latticeOut with monomers from mPolymerSystem */
    std::memset( mLatticeOut->host, 0, mLatticeOut->nBytes );
    for ( uint32_t t = 0; t < nAllMonomers; ++t )
    {
        mLatticeOut->host[ linearizeBoxVectorIndex( mPolymerSystem[ 4*t+0 ],
                                                    mPolymerSystem[ 4*t+1 ],
                                                    mPolymerSystem[ 4*t+2 ] ) ] = 1;
    }
    mLatticeOut->pushAsync();

    CUDA_ERROR( cudaGetDevice( &miGpuToUse ) );
    CUDA_ERROR( cudaGetDeviceProperties( &mCudaProps, miGpuToUse ) );
}


void UpdaterGPUScBFM_AB_Type::copyBondSet
( int dx, int dy, int dz, bool bondForbidden )
{
    mForbiddenBonds[ linearizeBondVectorIndex(dx,dy,dz) ] = bondForbidden;
}

void UpdaterGPUScBFM_AB_Type::setNrOfAllMonomers( uint32_t const rnAllMonomers )
{
    if ( this->nAllMonomers != 0 || mAttributeSystem != NULL ||
         mPolymerSystemSorted != NULL || mNeighborsSorted != NULL )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setNrOfAllMonomers] "
            << "Number of Monomers already set to " << nAllMonomers << "!\n"
            << "Or some arrays were already allocated "
            << "(mAttributeSystem=" << (void*) mAttributeSystem
            << ", mPolymerSystemSorted" << (void*) mPolymerSystemSorted
            << ", mNeighborsSorted" << (void*) mNeighborsSorted << ")\n";
        throw std::runtime_error( msg.str() );
    }

    this->nAllMonomers = rnAllMonomers;
    mAttributeSystem = new int32_t[ nAllMonomers ];
    if ( mAttributeSystem == NULL )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setNrOfAllMonomers] mAttributeSystem is still NULL after call to 'new int32_t[ " << nAllMonomers << " ]!\n";
        mLog( "Error" ) << msg.str();
        throw std::runtime_error( msg.str() );
    }
    mPolymerSystem.resize( nAllMonomers*4 );
    mNeighbors    .resize( nAllMonomers   );
    std::memset( &mNeighbors[0], 0, mNeighbors.size() * sizeof( mNeighbors[0] ) );
}

void UpdaterGPUScBFM_AB_Type::setPeriodicity
(
    bool const isPeriodicX,
    bool const isPeriodicY,
    bool const isPeriodicZ
)
{
    /* Compare inputs to hardcoded values. No ability yet to adjust dynamically */
#ifdef NONPERIODICITY
    if ( isPeriodicX || isPeriodicY || isPeriodicZ )
#else
    if ( ! isPeriodicX || ! isPeriodicY || ! isPeriodicZ )
#endif
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setPeriodicity" << "] "
            << "Simulation is intended to use completely "
        #ifdef NONPERIODICITY
            << "non-"
        #endif
            << "periodic boundary conditions, but setPeriodicity was called with "
            << "(" << isPeriodicX << "," << isPeriodicY << "," << isPeriodicZ << ")\n";
        mLog( "Error" ) << msg.str();
        throw std::invalid_argument( msg.str() );
    }
}

void UpdaterGPUScBFM_AB_Type::setAttribute( uint32_t i, int32_t attribute )
{
    mAttributeSystem[i] = attribute;
}

void UpdaterGPUScBFM_AB_Type::setMonomerCoordinates
(
    uint32_t const i,
    int32_t  const x,
    int32_t  const y,
    int32_t  const z
)
{
    mPolymerSystem.at( 4*i+0 ) = x;
    mPolymerSystem.at( 4*i+1 ) = y;
    mPolymerSystem.at( 4*i+2 ) = z;
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
    auto const iNew = mNeighbors[ iMonomer1 ].size++;
    if ( iNew > MAX_CONNECTIVITY-1 )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setConnectivity" << "] "
            << "The maximum amount of bonds per monomer (" << MAX_CONNECTIVITY
            << ") has been exceeded!\n";
        throw std::invalid_argument( msg.str() );
    }
    mNeighbors[ iMonomer1 ].neighborIds[ iNew ] = iMonomer2;
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
        auto const j = linearizeBoxVectorIndex( mPolymerSystem[ 4*i+0 ],
                                                mPolymerSystem[ 4*i+1 ],
                                                mPolymerSystem[ 4*i+2 ] );
        if ( j >= mBoxX * mBoxY * mBoxZ )
        {
            std::stringstream msg;
            msg
            << "[populateLattice] " << i << " -> ("
            << mPolymerSystem[ 4*i+0 ] << ","
            << mPolymerSystem[ 4*i+1 ] << ","
            << mPolymerSystem[ 4*i+2 ] << ") -> " << j << " is out of range "
            << "of (" << mBoxX << "," << mBoxY << "," << mBoxZ << ") = "
            << mBoxX * mBoxY * mBoxZ << "\n";
            throw std::runtime_error( msg.str() );
        }
        mLattice[ j ] = 1;
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
    if ( ! mLog.isActive( "Check" ) )
        return;

    if ( mLattice == NULL )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::checkSystem" << "] "
            << "mLattice is not allocated. You need to call "
            << "setNrOfAllMonomers and initialize before calling checkSystem!\n";
        mLog( "Error" ) << msg.str();
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
    for ( unsigned iNeighbor = 0; iNeighbor < mNeighbors[i].size; ++iNeighbor )
    {
        /* calculate the bond vector between the neighbor and this particle
         * neighbor - particle = ( dx, dy, dz ) */
        intCUDA * const neighbor = & mPolymerSystem[ 4*mNeighbors[i].neighborIds[ iNeighbor ] ];
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
                << mNeighbors[i].neighborIds[ iNeighbor ]+1 << " at ("
                << neighbor[0] << "," << neighbor[1] << "," << neighbor[2] << ")"
                << std::endl;
             throw std::runtime_error( msg.str() );
        }
    }
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
    this->destruct();

    cudaDeviceSynchronize();
    cudaProfilerStop();
}
