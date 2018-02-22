
__device__ uint32_t hash( uint32_t a )
{
    a = ( a + 0x7ed55d16 ) + ( a << 12 );
    a = ( a ^ 0xc761c23c ) ^ ( a >> 19 );
    a = ( a + 0x165667b1 ) + ( a << 5  );
    a = ( a + 0xd3a2646c ) ^ ( a << 9  );
    a = ( a + 0xfd7046c5 ) + ( a << 3  );
    a = ( a ^ 0xb55a4f09 ) ^ ( a >> 16 );
    return a;
}

__device__ inline uint32_t linearizeBoxVectorIndex
(
    uint32_t const & ix,
    uint32_t const & iy,
    uint32_t const & iz
)
{
    #if defined ( USE_ZCURVE_FOR_LATTICE )
        return   diluteBits< uint32_t, 2 >( ix & dcBoxXM1 )        +
               ( diluteBits< uint32_t, 2 >( iy & dcBoxYM1 ) << 1 ) +
               ( diluteBits< uint32_t, 2 >( iz & dcBoxZM1 ) << 2 );
    #else
        #if DEBUG_UPDATERGPUSCBFM_AB_TYPE > 10
            assert( isPowerOfTwo( dcBoxXM1 + 1 ) );
            assert( isPowerOfTwo( dcBoxYM1 + 1 ) );
            assert( isPowerOfTwo( dcBoxZM1 + 1 ) );
        #endif
        return   ( ix & dcBoxXM1 ) +
               ( ( iy & dcBoxYM1 ) << dcBoxXLog2  ) +
               ( ( iz & dcBoxZM1 ) << dcBoxXYLog2 );
    #endif
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
    uint32_t            const & x0        ,
    uint32_t            const & y0        ,
    uint32_t            const & z0        ,
    intCUDA             const & axis
)
{
    #if defined( USE_ZCURVE_FOR_LATTICE )
        auto const x0Abs  = diluteBits< uint32_t, 2 >( ( x0               ) & dcBoxXM1 );
        auto const x0PDX  = diluteBits< uint32_t, 2 >( ( x0 + uint32_t(1) ) & dcBoxXM1 );
        auto const x0MDX  = diluteBits< uint32_t, 2 >( ( x0 - uint32_t(1) ) & dcBoxXM1 );
        auto const y0Abs  = diluteBits< uint32_t, 2 >( ( y0               ) & dcBoxYM1 ) << 1;
        auto const y0PDY  = diluteBits< uint32_t, 2 >( ( y0 + uint32_t(1) ) & dcBoxYM1 ) << 1;
        auto const y0MDY  = diluteBits< uint32_t, 2 >( ( y0 - uint32_t(1) ) & dcBoxYM1 ) << 1;
        auto const z0Abs  = diluteBits< uint32_t, 2 >( ( z0               ) & dcBoxZM1 ) << 2;
        auto const z0PDZ  = diluteBits< uint32_t, 2 >( ( z0 + uint32_t(1) ) & dcBoxZM1 ) << 2;
        auto const z0MDZ  = diluteBits< uint32_t, 2 >( ( z0 - uint32_t(1) ) & dcBoxZM1 ) << 2;
    #else
        auto const x0Abs  =   ( x0               ) & dcBoxXM1;
        auto const x0PDX  =   ( x0 + uint32_t(1) ) & dcBoxXM1;
        auto const x0MDX  =   ( x0 - uint32_t(1) ) & dcBoxXM1;
        auto const y0Abs  = ( ( y0               ) & dcBoxYM1 ) << dcBoxXLog2;
        auto const y0PDY  = ( ( y0 + uint32_t(1) ) & dcBoxYM1 ) << dcBoxXLog2;
        auto const y0MDY  = ( ( y0 - uint32_t(1) ) & dcBoxYM1 ) << dcBoxXLog2;
        auto const z0Abs  = ( ( z0               ) & dcBoxZM1 ) << dcBoxXYLog2;
        auto const z0PDZ  = ( ( z0 + uint32_t(1) ) & dcBoxZM1 ) << dcBoxXYLog2;
        auto const z0MDZ  = ( ( z0 - uint32_t(1) ) & dcBoxZM1 ) << dcBoxXYLog2;
    #endif

    auto const dx = DXTable_d[ axis ];   // 2*axis-1
    auto const dy = DYTable_d[ axis ];   // 2*(axis&1)-1
    auto const dz = DZTable_d[ axis ];   // 2*(axis&1)-1

    uint32_t is[9];

    #if defined( USE_ZCURVE_FOR_LATTICE )
        switch ( axis >> intCUDA(1) )
        {
            case 0: is[7] = ( x0 + decltype(dx)(2) * dx ) & dcBoxXM1; break;
            case 1: is[7] = ( y0 + decltype(dy)(2) * dy ) & dcBoxYM1; break;
            case 2: is[7] = ( z0 + decltype(dz)(2) * dz ) & dcBoxZM1; break;
        }
        is[7] = diluteBits< uint32_t, 2 >( is[7] ) << ( axis >> intCUDA(1) );
    #else
        switch ( axis >> intCUDA(1) )
        {
            case 0: is[7] =   ( x0 + decltype(dx)(2) * dx ) & dcBoxXM1; break;
            case 1: is[7] = ( ( y0 + decltype(dy)(2) * dy ) & dcBoxYM1 ) << dcBoxXLog2; break;
            case 2: is[7] = ( ( z0 + decltype(dz)(2) * dz ) & dcBoxZM1 ) << dcBoxXYLog2; break;
        }
    #endif
    switch ( axis >> intCUDA(1) )
    {
        case 0: //-+x
        {
            is[2]  = is[7] | z0Abs;
            is[5]  = is[7] | z0MDZ;
            is[8]  = is[7] | z0PDZ;
            is[0]  = is[2] | y0MDY;
            is[1]  = is[2] | y0Abs;
            is[2] |=         y0PDY;
            is[3]  = is[5] | y0MDY;
            is[4]  = is[5] | y0Abs;
            is[5] |=         y0PDY;
            is[6]  = is[8] | y0MDY;
            is[7]  = is[8] | y0Abs;
            is[8] |=         y0PDY;
            break;
        }
        case 1: //-+y
        {
            is[2]  = is[7] | z0MDZ;
            is[5]  = is[7] | z0Abs;
            is[8]  = is[7] | z0PDZ;
            is[0]  = is[2] | x0MDX;
            is[1]  = is[2] | x0Abs;
            is[2] |=         x0PDX;
            is[3]  = is[5] | x0MDX;
            is[4]  = is[5] | x0Abs;
            is[5] |=         x0PDX;
            is[6]  = is[8] | x0MDX;
            is[7]  = is[8] | x0Abs;
            is[8] |=         x0PDX;
            break;
        }
        case 2: //-+z
        {
            is[2]  = is[7] | y0MDY;
            is[5]  = is[7] | y0Abs;
            is[8]  = is[7] | y0PDY;
            is[0]  = is[2] | x0MDX;
            is[1]  = is[2] | x0Abs;
            is[2] |=         x0PDX;
            is[3]  = is[5] | x0MDX;
            is[4]  = is[5] | x0Abs;
            is[5] |=         x0PDX;
            is[6]  = is[8] | x0MDX;
            is[7]  = is[8] | x0Abs;
            is[8] |=         x0PDX;
            break;
        }
    }
    bool const isOccupied = tex1Dfetch< uint8_t >( texLattice, is[0] ) |
                            tex1Dfetch< uint8_t >( texLattice, is[1] ) |
                            tex1Dfetch< uint8_t >( texLattice, is[2] ) |
                            tex1Dfetch< uint8_t >( texLattice, is[3] ) |
                            tex1Dfetch< uint8_t >( texLattice, is[4] ) |
                            tex1Dfetch< uint8_t >( texLattice, is[5] ) |
                            tex1Dfetch< uint8_t >( texLattice, is[6] ) |
                            tex1Dfetch< uint8_t >( texLattice, is[7] ) |
                            tex1Dfetch< uint8_t >( texLattice, is[8] );
    return isOccupied;
}

#ifdef USE_BIT_PACKING
__device__ inline bool checkFrontBitPacked
(
    cudaTextureObject_t const & texLattice,
    uint32_t            const & x0        ,
    uint32_t            const & y0        ,
    uint32_t            const & z0        ,
    intCUDA             const & axis
)
{
    auto const x0Abs  = diluteBits< uint32_t, 2 >( ( x0               ) & dcBoxXM1 );
    auto const x0PDX  = diluteBits< uint32_t, 2 >( ( x0 + uint32_t(1) ) & dcBoxXM1 );
    auto const x0MDX  = diluteBits< uint32_t, 2 >( ( x0 - uint32_t(1) ) & dcBoxXM1 );
    auto const y0Abs  = diluteBits< uint32_t, 2 >( ( y0               ) & dcBoxYM1 ) << 1;
    auto const y0PDY  = diluteBits< uint32_t, 2 >( ( y0 + uint32_t(1) ) & dcBoxYM1 ) << 1;
    auto const y0MDY  = diluteBits< uint32_t, 2 >( ( y0 - uint32_t(1) ) & dcBoxYM1 ) << 1;
    auto const z0Abs  = diluteBits< uint32_t, 2 >( ( z0               ) & dcBoxZM1 ) << 2;
    auto const z0PDZ  = diluteBits< uint32_t, 2 >( ( z0 + uint32_t(1) ) & dcBoxZM1 ) << 2;
    auto const z0MDZ  = diluteBits< uint32_t, 2 >( ( z0 - uint32_t(1) ) & dcBoxZM1 ) << 2;

    auto const dx = DXTable_d[ axis ];   // 2*axis-1
    auto const dy = DYTable_d[ axis ];   // 2*(axis&1)-1
    auto const dz = DZTable_d[ axis ];   // 2*(axis&1)-1

    uint32_t is[9];
    switch ( axis >> intCUDA(1) )
    {
        case 0: is[7] = ( x0 + decltype(dx)(2) * dx ) & dcBoxXM1; break;
        case 1: is[7] = ( y0 + decltype(dy)(2) * dy ) & dcBoxYM1; break;
        case 2: is[7] = ( z0 + decltype(dz)(2) * dz ) & dcBoxZM1; break;
    }
    is[7] = diluteBits< uint32_t, 2 >( is[7] ) << ( axis >> intCUDA(1) );
    switch ( axis >> intCUDA(1) )
    {
        case 0: //-+x
            is[2]  = is[7] + z0Abs; is[5]  = is[7] + z0MDZ; is[8]  = is[7] + z0PDZ;
            is[0]  = is[2] + y0MDY; is[1]  = is[2] + y0Abs; is[2] +=         y0PDY;
            is[3]  = is[5] + y0MDY; is[4]  = is[5] + y0Abs; is[5] +=         y0PDY;
            is[6]  = is[8] + y0MDY; is[7]  = is[8] + y0Abs; is[8] +=         y0PDY;
            break;
        case 1: //-+y
            is[2]  = is[7] + z0MDZ; is[5]  = is[7] + z0Abs; is[8]  = is[7] + z0PDZ;
            is[0]  = is[2] + x0MDX; is[1]  = is[2] + x0Abs; is[2] +=         x0PDX;
            is[3]  = is[5] + x0MDX; is[4]  = is[5] + x0Abs; is[5] +=         x0PDX;
            is[6]  = is[8] + x0MDX; is[7]  = is[8] + x0Abs; is[8] +=         x0PDX;
            break;
        case 2: //-+z
            is[2]  = is[7] + y0MDY; is[5]  = is[7] + y0Abs; is[8]  = is[7] + y0PDY;
            is[0]  = is[2] + x0MDX; is[1]  = is[2] + x0Abs; is[2] +=         x0PDX;
            is[3]  = is[5] + x0MDX; is[4]  = is[5] + x0Abs; is[5] +=         x0PDX;
            is[6]  = is[8] + x0MDX; is[7]  = is[8] + x0Abs; is[8] +=         x0PDX;
            break;
    }
    return bitPackedTextureGet< uint8_t >( texLattice, is[0] ) +
           bitPackedTextureGet< uint8_t >( texLattice, is[1] ) +
           bitPackedTextureGet< uint8_t >( texLattice, is[2] ) +
           bitPackedTextureGet< uint8_t >( texLattice, is[3] ) +
           bitPackedTextureGet< uint8_t >( texLattice, is[4] ) +
           bitPackedTextureGet< uint8_t >( texLattice, is[5] ) +
           bitPackedTextureGet< uint8_t >( texLattice, is[6] ) +
           bitPackedTextureGet< uint8_t >( texLattice, is[7] ) +
           bitPackedTextureGet< uint8_t >( texLattice, is[8] );
}
#endif

__device__ __host__ inline uintCUDA linearizeBondVectorIndex
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
    //assert( -4 <= x && x <= 4 );
    //assert( -4 <= y && y <= 4 );
    //assert( -4 <= z && z <= 4 );
    return   ( x & intCUDA(7) /* 0b111 */ ) +
           ( ( y & intCUDA(7) /* 0b111 */ ) << intCUDA(3) ) +
           ( ( z & intCUDA(7) /* 0b111 */ ) << intCUDA(6) );
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
 * @param[in] iOffset Offste to curent species we are supposed to work on.
 *            We can't simply adjust dpPolymerSystem before calling the kernel,
 *            because we are accessing the neighbors, therefore need all the
 *            polymer data, especially for other species.
 *
 * Note: all of the three kernels do quite few work. They basically just fetch
 *       data, and check one condition and write out again. There isn't even
 *       a loop and most of the work seems to be boiler plate initialization
 *       code which could be cut if the kernels could be merged together.
 *       Why are there three kernels instead of just one
 *        -> for global synchronization
 */
__global__ void kernelSimulationScBFMCheckSpecies
(
    vecIntCUDA  const * const __restrict__ dpPolymerSystem         ,
    T_Flags           * const __restrict__ dpPolymerFlags          ,
    uint32_t            const              iOffset                 ,
    uint8_t           * const __restrict__ dpLatticeTmp            ,
    uint32_t    const * const __restrict__ dpNeighbors             ,
    uint32_t            const              rNeighborsPitchElements ,
    uint8_t     const * const __restrict__ dpNeighborsSizes        ,
    uint32_t            const              nMonomers               ,
    uint32_t            const              rSeed                   ,
    cudaTextureObject_t const              texLatticeRefOut
)
{
    uint32_t rn;
    int iGrid = 0;
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x, ++iGrid )
    {
        auto const r0 = dpPolymerSystem[ iOffset + iMonomer ];
        /* upcast int3 to int4 in preparation to use PTX SIMD instructions */
        //int4 const r0 = { r0Raw.x, r0Raw.y, r0Raw.z, 0 }; // not faster nor slower
        //select random direction. Own implementation of an rng :S? But I think it at least# was initialized using the LeMonADE RNG ...
        if ( iGrid % 1 == 0 ) // 12 = floor( log(2^32) / log(6) )
            rn = hash( hash( iMonomer ) ^ rSeed );
        T_Flags const direction = rn % T_Flags(6); rn /= T_Flags(6);
        T_Flags properties = 0;

         /* select random direction. Do this with bitmasking instead of lookup ??? */
        /* int4 const dr = { DXTable_d[ direction ],
                          DYTable_d[ direction ],
                          DZTable_d[ direction ], 0 }; */
        uint3 const r1 = { r0.x + DXTable_d[ direction ],
                           r0.y + DYTable_d[ direction ],
                           r0.z + DZTable_d[ direction ] };

    #ifdef NONPERIODICITY
       /* check whether the new location of the particle would be inside the box
        * if the box is not periodic, if not, then don't move the particle */
        if ( uint32_t(0) <= r1.x && r1.x < dcBoxXM1 &&
             uint32_t(0) <= r1.y && r1.y < dcBoxYM1 &&
             uint32_t(0) <= r1.z && r1.z < dcBoxZM1    )
        {
    #endif
            /* check whether the new position would result in invalid bonds
             * between this monomer and its neighbors */
            auto const nNeighbors = dpNeighborsSizes[ iOffset + iMonomer ];
            bool forbiddenBond = false;
            for ( auto iNeighbor = decltype( nNeighbors )(0); iNeighbor < nNeighbors; ++iNeighbor )
            {
                auto const iGlobalNeighbor = dpNeighbors[ iNeighbor * rNeighborsPitchElements + iMonomer ];
                auto const data2 = dpPolymerSystem[ iGlobalNeighbor ];
                if ( dpForbiddenBonds[ linearizeBondVectorIndex( data2.x - r1.x, data2.y - r1.y, data2.z - r1.z ) ] )
                {
                    forbiddenBond = true;
                    break;
                }
            }
            if ( ! forbiddenBond && ! checkFront( texLatticeRefOut, r0.x, r0.y, r0.z, direction ) )
            {
                /* everything fits so perform move on temporary lattice */
                /* can I do this ??? dpPolymerSystem is the device pointer to the read-only
                 * texture used above. Won't this result in read-after-write race-conditions?
                 * Then again the written / changed bits are never used in the above code ... */
                properties = ( direction << T_Flags(2) ) + T_Flags(1) /* can-move-flag */;
            #ifdef USE_BIT_PACKING_TMP_LATTICE
                bitPackedSet( dpLatticeTmp, linearizeBoxVectorIndex( r1.x, r1.y, r1.z ) );
            #else
                dpLatticeTmp[ linearizeBoxVectorIndex( r1.x, r1.y, r1.z ) ] = 1;
            #endif
            }
    #ifdef NONPERIODICITY
        }
    #endif
        dpPolymerFlags[ iOffset + iMonomer ] = properties;
    }
}

__global__ void kernelSimulationScBFMPerformSpeciesAndApply
(
    vecIntCUDA          * const __restrict__ dpPolymerSystem,
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

        auto const r0 = ( (CudaVec4< intCUDA >::value_type *) dpPolymerSystem )[ iMonomer ];
        auto const direction = ( properties >> T_Flags(2) ) & T_Flags(7); // 7=0b111
    #ifdef USE_BIT_PACKING_TMP_LATTICE
        if ( checkFrontBitPacked( texLatticeTmp, r0.x, r0.y, r0.z, direction ) )
    #else
        if ( checkFront( texLatticeTmp, r0.x, r0.y, r0.z, direction ) )
    #endif
            continue;

        CudaVec4< intCUDA >::value_type const r1 = {
            intCUDA( r0.x + DXTableIntCUDA_d[ direction ] ),
            intCUDA( r0.y + DYTableIntCUDA_d[ direction ] ),
            intCUDA( r0.z + DZTableIntCUDA_d[ direction ] ), 0
        };
        /* If possible, perform move now on normal lattice */
        dpLattice[ linearizeBoxVectorIndex( r0.x, r0.y, r0.z ) ] = 0;
        dpLattice[ linearizeBoxVectorIndex( r1.x, r1.y, r1.z ) ] = 1;
        dpPolymerSystem[ iMonomer ] = r1;
    }
}


void UpdaterGPUScBFM_AB_Type::runSimulationOnGPU
(
    int32_t const nMonteCarloSteps
)
{
    std::clock_t const t0 = std::clock();

    auto const nSpecies = mnElementsInGroup.size();

    /**
     * Statistics (min, max, mean, stddev) on filtering. Filtered because of:
     *   0: bonds, 1: volume exclusion, 2: volume exclusion (parallel)
     * These statistics are done for each group separately
     */
    std::vector< std::vector< double > > sums, sums2, mins, maxs, ns;
    std::vector< unsigned long long int > vFiltered;
    unsigned long long int * dpFiltered = NULL;
    auto constexpr nFilters = 5;
    if ( mLog.isActive( "Stats" ) )
    {
        sums .resize( nSpecies, std::vector< double >( nFilters, 0            ) );
        sums2.resize( nSpecies, std::vector< double >( nFilters, 0            ) );
        mins .resize( nSpecies, std::vector< double >( nFilters, nAllMonomers ) );
        maxs .resize( nSpecies, std::vector< double >( nFilters, 0            ) );
        ns   .resize( nSpecies, std::vector< double >( nFilters, 0            ) );
        /* ns needed because we need to know how often each group was advanced */
        vFiltered.resize( nFilters );
        CUDA_ERROR( cudaMalloc( &dpFiltered, nFilters * sizeof( *dpFiltered ) ) );
        CUDA_ERROR( cudaMemsetAsync( (void*) dpFiltered, 0, nFilters * sizeof( *dpFiltered ), mStream ) );
    }

    /**
     * Logic for determining the best threadsPerBlock configuration
     *
     * This might be dependent on the species, therefore for each species
     * store the current best thread count and all timings.
     * As the cudaEventSynchronize timings are expensive, stop benchmarking
     * after having found the best configuration.
     * Only try out power two multiples of warpSize up to maxThreadsPerBlock,
     * e.g. 32, 64, 128, 256, 512, 1024, because smaller than warp size
     * should never lead to a speedup and non power of twos, e.g. 196 even,
     * won't be able to perfectly fill out the shared multi processor.
     * Also, automatically determine whether cudaMemset is faster or not (after
     * we found the best threads per block configuration
     * note: test example best configuration was 128 threads per block and use
     *       the cudaMemset version instead of the third kernel
     */
    std::vector< int > vnThreadsToTry;
    for ( auto nThreads = mCudaProps.warpSize; nThreads <= mCudaProps.maxThreadsPerBlock; nThreads *= 2 )
        vnThreadsToTry.push_back( nThreads );
    assert( vnThreadsToTry.size() > 0 );
    struct SpeciesBenchmarkData
    {
        /* 2 vectors of double for measurements with and without cudaMemset */
        std::vector< std::vector< float > > timings;
        std::vector< std::vector< int   > > nRepeatTimings;
        int iBestThreadCount;
        bool useCudaMemset;
        bool decidedAboutThreadCount;
        bool decidedAboutCudaMemset;
    };
    std::vector< SpeciesBenchmarkData > benchmarkInfo( nSpecies, SpeciesBenchmarkData{
        std::vector< std::vector< float > >( 2 /* true or false */,
            std::vector< float >( vnThreadsToTry.size(),
                std::numeric_limits< float >::infinity() ) ),
        std::vector< std::vector< int   > >( 2 /* true or false */,
            std::vector< int   >( vnThreadsToTry.size(),
            2 /* repeat 2 time, i.e. execute three times */ ) ),
#ifdef AUTO_CONFIGURE_BEST_SETTINGS_FOR_PSCBFM_ALGORITHM
        0, true, vnThreadsToTry.size() <= 1, false
#else
        2, true, true, true
#endif
    } );
    cudaEvent_t tOneGpuLoop0, tOneGpuLoop1;
    cudaEventCreate( &tOneGpuLoop0 );
    cudaEventCreate( &tOneGpuLoop1 );

    cudaEvent_t tGpu0, tGpu1;
    if ( mLog.isActive( "Benchmark" ) )
    {
        cudaEventCreate( &tGpu0 );
        cudaEventCreate( &tGpu1 );
        cudaEventRecord( tGpu0, mStream );
    }

    /* run simulation */
    for ( int32_t iStep = 1; iStep <= nMonteCarloSteps; ++iStep )
    {
        /* one Monte-Carlo step:
         *  - tries to move on average all particles one time
         *  - each particle could be touched, not just one group */
        for ( uint32_t iSubStep = 0; iSubStep < nSpecies; ++iSubStep )
        {
            /* randomly choose which monomer group to advance */
            auto const iSpecies = randomNumbers.r250_rand32() % nSpecies;
            auto const seed     = randomNumbers.r250_rand32();
            auto const nThreads = vnThreadsToTry.at( benchmarkInfo[ iSpecies ].iBestThreadCount );
            auto const nBlocks  = ceilDiv( mnElementsInGroup[ iSpecies ], nThreads );
            auto const needToBenchmark = ! (
                benchmarkInfo[ iSpecies ].decidedAboutThreadCount &&
                benchmarkInfo[ iSpecies ].decidedAboutCudaMemset );
            auto const useCudaMemset = benchmarkInfo[ iSpecies ].useCudaMemset;
            if ( needToBenchmark )
                cudaEventRecord( tOneGpuLoop0, mStream );

            /*
            if ( iStep < 3 )
                mLog( "Info" ) << "Calling Check-Kernel for species " << iSpecies << " for uint32_t * " << (void*) mNeighborsSorted->gpu << " + " << mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ) << " = " << (void*)( mNeighborsSorted->gpu + mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ) ) << " with pitch " << mNeighborsSortedInfo.getMatrixPitchElements( iSpecies ) << "\n";
            */

            kernelSimulationScBFMCheckSpecies
            <<< nBlocks, nThreads, 0, mStream >>>(
                mPolymerSystemSorted->gpu,
                mPolymerFlags->gpu,
                iSubGroupOffset[ iSpecies ],
                mLatticeTmp->gpu,
                mNeighborsSorted->gpu + mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ),
                mNeighborsSortedInfo.getMatrixPitchElements( iSpecies ),
                mNeighborsSortedSizes->gpu,
                mnElementsInGroup[ iSpecies ], seed,
                mLatticeOut->texture
            );

            if ( useCudaMemset )
            {
                kernelSimulationScBFMPerformSpeciesAndApply
                <<< nBlocks, nThreads, 0, mStream >>>(
                    mPolymerSystemSorted->gpu + iSubGroupOffset[ iSpecies ],
                    mPolymerFlags->gpu + iSubGroupOffset[ iSpecies ],
                    mLatticeOut->gpu,
                    mnElementsInGroup[ iSpecies ],
                    mLatticeTmp->texture
                );
            }
            else
            {
                kernelSimulationScBFMPerformSpecies
                <<< nBlocks, nThreads, 0, mStream >>>(
                    mPolymerSystemSorted->gpu + iSubGroupOffset[ iSpecies ],
                    mPolymerFlags->gpu + iSubGroupOffset[ iSpecies ],
                    mLatticeOut->gpu,
                    mnElementsInGroup[ iSpecies ],
                    mLatticeTmp->texture
                );
            }

            if ( useCudaMemset )
            {
                #ifdef USE_THRUST_FILL
                    thrust::fill( thrust::system::cuda::par, (uint64_t*)  mLatticeTmp->gpu,
                                  (uint64_t*)( mLatticeTmp->gpu + mLatticeTmp->nElements ), 0 );
                #else
                    #ifdef USE_BIT_PACKING_TMP_LATTICE
                        cudaMemsetAsync( (void*) mLatticeTmp->gpu, 0, mLatticeTmp->nBytes / CHAR_BIT, mStream );
                    #else
                        cudaMemsetAsync( (void*) mLatticeTmp->gpu, 0, mLatticeTmp->nBytes, mStream );
                    #endif
                #endif
            }
            else
            {
                kernelSimulationScBFMZeroArraySpecies
                <<< nBlocks, nThreads, 0, mStream >>>(
                    mPolymerSystemSorted->gpu + iSubGroupOffset[ iSpecies ],
                    mPolymerFlags->gpu + iSubGroupOffset[ iSpecies ],
                    mLatticeTmp->gpu,
                    mnElementsInGroup[ iSpecies ]
                );
            }

            if ( needToBenchmark )
            {
                auto & info = benchmarkInfo[ iSpecies ];
                cudaEventRecord( tOneGpuLoop1, mStream );
                cudaEventSynchronize( tOneGpuLoop1 );
                float milliseconds = 0;
                cudaEventElapsedTime( & milliseconds, tOneGpuLoop0, tOneGpuLoop1 );
                auto const iThreadCount = info.iBestThreadCount;
                auto & oldTiming = info.timings.at( useCudaMemset ).at( iThreadCount );
                oldTiming = std::min( oldTiming, milliseconds );

                mLog( "Info" )
                << "Using " << nThreads << " threads (" << nBlocks << " blocks)"
                << " and using " << ( useCudaMemset ? "cudaMemset" : "kernelZeroArray" )
                << " for species " << iSpecies << " took " << milliseconds << "ms\n";

                auto & nStillToRepeat = info.nRepeatTimings.at( useCudaMemset ).at( iThreadCount );
                if ( nStillToRepeat > 0 )
                    --nStillToRepeat;
                else if ( ! info.decidedAboutThreadCount )
                {
                    /* if not the first timing, then decide whether we got slower,
                     * i.e. whether we found the minimum in the last step and
                     * have to roll back */
                    if ( iThreadCount > 0 )
                    {
                        if ( info.timings.at( useCudaMemset ).at( iThreadCount-1 ) < milliseconds )
                        {
                            --info.iBestThreadCount;
                            info.decidedAboutThreadCount = true;
                        }
                        else
                            ++info.iBestThreadCount;
                    }
                    else
                        ++info.iBestThreadCount;
                    /* if we can't increment anymore, then we are finished */
                    assert( (size_t) info.iBestThreadCount <= vnThreadsToTry.size() );
                    if ( (size_t) info.iBestThreadCount == vnThreadsToTry.size() )
                    {
                        --info.iBestThreadCount;
                        info.decidedAboutThreadCount = true;
                    }
                    if ( info.decidedAboutThreadCount )
                    {
                        /* then in the next term try out changing cudaMemset
                         * version to custom kernel version (or vice-versa) */
                        if ( ! info.decidedAboutCudaMemset )
                            info.useCudaMemset = ! info.useCudaMemset;
                        mLog( "Info" )
                        << "Using " << vnThreadsToTry.at( info.iBestThreadCount )
                        << " threads per block is the fastest for species "
                        << iSpecies << ".\n";
                    }
                }
                else if ( ! info.decidedAboutCudaMemset )
                {
                    info.decidedAboutCudaMemset = true;
                    if ( info.timings.at( ! useCudaMemset ).at( iThreadCount ) < milliseconds )
                        info.useCudaMemset = ! useCudaMemset;
                    if ( info.decidedAboutCudaMemset )
                    {
                        mLog( "Info" )
                        << "Using " << ( info.useCudaMemset ? "cudaMemset" : "kernelZeroArray" )
                        << " is the fastest for species " << iSpecies << ".\n";
                    }
                }
            }
        } // iSubstep
    } // iStep

    if ( mLog.isActive( "Benchmark" ) )
    {
        // https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/#disqus_thread
        cudaEventRecord( tGpu1, mStream );
        cudaEventSynchronize( tGpu1 );  // basically a StreamSynchronize
        float milliseconds = 0;
        cudaEventElapsedTime( & milliseconds, tGpu0, tGpu1 );
        std::stringstream sBuffered;
        sBuffered << "tGpuLoop = " << milliseconds / 1000. << "s\n";
        mLog( "Benchmark" ) << sBuffered.str();
    }

    mtCopyBack0 = std::chrono::high_resolution_clock::now();

    /* copy into mPolymerSystem and drop the property tag while doing so.
     * would be easier and probably more efficient if mPolymerSystem->gpu/host
     * would be a struct of arrays instead of an array of structs !!! */
    mPolymerSystemSorted->pop( false ); // sync

    if ( mLog.isActive( "Benchmark" ) )
    {
        std::clock_t const t1 = std::clock();
        double const dt = float(t1-t0) / CLOCKS_PER_SEC;
        mLog( "Benchmark" )
        << "run time (GPU): " << nMonteCarloSteps << "\n"
        << "mcs = " << nMonteCarloSteps  << "  speed [performed monomer try and move/s] = MCS*N/t: "
        << nMonteCarloSteps * ( nAllMonomers / dt )  << "     runtime[s]:" << dt << "\n";
    }

    /* untangle reordered array so that LeMonADE can use it again */
    for ( size_t i = 0u; i < nAllMonomers; ++i )
    {
        auto const pTarget = mPolymerSystemSorted->host + iToiNew[i];
        if ( i < 10 )
            mLog( "Info" ) << "Copying back " << i << " from " << iToiNew[i] << "\n";
        mPolymerSystem[ 4*i+0 ] = pTarget->x;
        mPolymerSystem[ 4*i+1 ] = pTarget->y;
        mPolymerSystem[ 4*i+2 ] = pTarget->z;
        mPolymerSystem[ 4*i+3 ] = pTarget->w;
    }

    checkSystem(); // no-op if "Check"-level deactivated
}
