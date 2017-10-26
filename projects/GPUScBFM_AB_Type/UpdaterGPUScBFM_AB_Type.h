/*
 * UpdaterGPUScBFM_AB_Type.h
 *
 *  Created on: 27.07.2017
 *      Author: Ron Dockhorn
 */

#pragma once


#include <cassert>
#include <cstdio>                           // printf
#include <stdint.h>                         // uint32_t

#include <cuda.h>

#include <LeMonADE/utility/RandomNumberGenerators.h>


#define __FILENAME__ (__builtin_strrchr(__FILE__, '/') ? __builtin_strrchr(__FILE__, '/') + 1 : __FILE__)

#define CUDA_CHECK(cmd)                                                 \
{                                                                       \
    cudaError_t error = cmd;                                            \
    if ( error != cudaSuccess )                                         \
    {                                                                   \
        printf( "<%s>:%i ", __FILENAME__, __LINE__ );                       \
        printf( "[CUDA] Error: %s\n", cudaGetErrorString( error ) );    \
        exit(1);                                                        \
    }                                                                   \
}

/* hard-coded optimal CUDA Kernel parameters */
#define NUMBLOCKS 16
#define NUMTHREADS 256

/* still used ??? Will this need a switch? */
#if false
    typedef uint32_t uintCUDA;
    typedef int32_t  intCUDA;
    #define MASK5BITS 0x7FFFFFE0
#else
    typedef uint16_t uintCUDA;
    typedef int16_t  intCUDA;
    #define MASK5BITS 0x7FE0
#endif


#define MAX_CONNECTIVITY 4

// #define NONPERIODICITY

struct MonoInfo
{
    uint32_t size;
    uint32_t bondsMonomerIdx[ MAX_CONNECTIVITY ];
};



class UpdaterGPUScBFM_AB_Type
{
private:
    RandomNumberGenerators randomNumbers;

    bool mForbiddenBonds[512];
    //int BondAsciiArray[512];

    uint32_t   nAllMonomers       ;

    uint32_t   nStars             ;
    uint32_t   nMonomersPerStarArm;
    uint32_t   nCrosslinker       ;

    uint8_t *  mLattice           ;
    int32_t *  mPolymerSystem     ;
    int32_t *  mAttributeSystem   ;

    //! Holds connectivity information
    struct MonoNNIndex
    {
        uint32_t size;
        uint32_t bondsMonomerIdx[ MAX_CONNECTIVITY ];
    };

    MonoNNIndex **monosNNidx;

    uint32_t   Box_X;
    uint32_t   Box_Y;
    uint32_t   Box_Z;

    uint32_t   Box_XM1;
    uint32_t   Box_YM1;
    uint32_t   Box_ZM1;

    uint32_t   Box_XPRO;
    uint32_t   Box_PROXY;

    intCUDA  * mPolymerSystem_device;
    intCUDA  * mPolymerSystem_host;

    uint8_t  * mLatticeOut_host;
    uint8_t  * mLatticeOut_device;

    uint8_t  * mLatticeTmp_device;
    uint8_t  * mLatticeTmp_host;

    MonoInfo * MonoInfo_host, *MonoInfo_device;   // Pointer to host & device arrays of structure

    uint32_t * MonomersSpeziesIdx_A_host;
    uint32_t * MonomersSpeziesIdx_B_host;

    uint32_t * MonomersSpeziesIdx_A_device;
    uint32_t * MonomersSpeziesIdx_B_device;

    uint32_t   numblocksSpecies_A;
    uint32_t   numblocksSpecies_B;


    /**
     * Packs the three given coordinates into 9 bits. Note that 2^9=512. This
     * explains the forbiddenBonds table being 512 entries large!
     */
    int IndexBondArray( int const x, int const y, int const z );

    void checkSystem();

public:
    UpdaterGPUScBFM_AB_Type(){};
    virtual ~UpdaterGPUScBFM_AB_Type();

    void initialize( int iGpuToUse );
    inline bool execute(){ return true; }
    void cleanup();

    /* setter methods */
    inline void copyBondSet( int dx, int dy, int dz, bool bondForbidden ){ mForbiddenBonds[ IndexBondArray(dx,dy,dz) ] = bondForbidden; }
    inline void setAttribute( uint32_t i, int32_t attribute ){ mAttributeSystem[i] = attribute; }
    void setNrOfAllMonomers( uint32_t nAllMonomers );
    void setNetworkIngredients( uint32_t numPEG, uint32_t numPEGArm, uint32_t numCL );
    inline void setMonomerCoordinates( uint32_t i, int32_t x, int32_t y, int32_t z )
    {
        mPolymerSystem[ 3*i+0 ] = x;
        mPolymerSystem[ 3*i+1 ] = y;
        mPolymerSystem[ 3*i+2 ] = z;
    }
    void setConnectivity( uint32_t monoidx1, uint32_t monoidx2 );
    void setLatticeSize( uint32_t boxX, uint32_t boxY, uint32_t boxZ );

    void populateLattice();
    void runSimulationOnGPU( int32_t nrMCS_per_Call );

    inline int32_t getMonomerPositionInX( uint32_t i ){ return mPolymerSystem[3*i+0]; }
    inline int32_t getMonomerPositionInY( uint32_t i ){ return mPolymerSystem[3*i+1]; }
    inline int32_t getMonomerPositionInZ( uint32_t i ){ return mPolymerSystem[3*i+2]; }

    void setPeriodicity( bool isPeriodicX, bool isPeriodicY, bool isPeriodicZ );

};