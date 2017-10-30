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

#include <LeMonADE/utility/RandomNumberGenerators.h>

#include "cudacommon.hpp"



/* This is still used, at least the 32 bit version! */
#if 0
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

    /**
     * Vector of length boxX * boxY * boxZ. Actually only contains 0 if
     * the cell / lattice point is empty or 1 if it is occupied by a monomer
     * Suggestion: bitpack it to save 8 times memory and possibly make the
     *             the reading faster if it is memory bound ???
     */
    uint8_t * mLattice;
    MirroredTexture< uint8_t > * mLatticeOut, * mLatticeTmp;

    /* copy into mPolymerSystem and drop the property tag while doing so.
     * would be easier and probably more efficient if mPolymerSystem_device/host
     * would be a struct of arrays instead of an array of structs !!! */
    /**
     * Contains the nMonomers particles as well as a property tag for each:
     *   [ x0, y0, z0, p0, x1, y1, z1, p1, ... ]
     * The property tags p are bit packed:
     *                        8  7  6  5  4  3  2  1  0
     * +--------+--+--+--+--+--+--+--+--+--+--+--+--+--+
     * | unused |  |  |  |  |c |   nnr  |  dir   |move |
     * +--------+--+--+--+--+--+--+--+--+--+--+--+--+--+
     *  c   ... charged: 0 no, 1: yes
     *  nnr ... number of neighbors, this will get populated from LeMonADE's
     *          get get
     *  move ... Bit 0 set by kernelCheckSpecies if move was found to be possible
     *           Bit 1 set by kernelPerformSpecies if move still possible
     *           heeding the new locations of all particles.
     *           If both bits set, the move gets applied to polymerSystem in
     *           kernelZeroArrays
     * The saved location is used as the lower left front corner when
     * populating the lattice with 2x2x2 boxes representing the monomers
     */
    intCUDA * mPolymerSystem;
    intCUDA * mPolymerSystem_device;
    //MirroredTexture< intCUDA > * mPolymerSystem;

    int32_t *  mAttributeSystem;

    /* stores amount and IDs of neighbors for each monomer */
public:
    struct MonomerEdges
    {
        uint32_t size;
        uint32_t neighborIds[ MAX_CONNECTIVITY ];
    };
private:
    MonomerEdges * mNeighbors;
    MonomerEdges * MonoInfo_host, *MonoInfo_device;

    uint32_t   mBoxX     ;
    uint32_t   mBoxY     ;
    uint32_t   mBoxZ     ;
    uint32_t   mBoxXM1   ;
    uint32_t   mBoxYM1   ;
    uint32_t   mBoxZM1   ;
    uint32_t   mBoxXLog2 ;
    uint32_t   mBoxXYLog2;

    /**
     * These are arrays containing the monomer indices for the respective
     * species (sorted ascending). E.g. for AABABBA this would be:
     *   mMonomerIdsA = { 0,1,3,6 } -> nMonomersSpeciesA = 4
     *   mMonomerIdsB = { 1,4,5 }   -> nMonomersSpeciesB = 3
     */
    uint32_t nMonomersSpeciesA, nMonomersSpeciesB;
    MirroredTexture< uint32_t > * mMonomerIdsA, * mMonomerIdsB;

    uint32_t linearizeBoxVectorIndex
    (
        uint32_t const & ix,
        uint32_t const & iy,
        uint32_t const & iz
    );

    /**
     * Checks for excluded volume condition and for correctness of all monomer bonds
     */
    void checkSystem();

public:
    UpdaterGPUScBFM_AB_Type();
    virtual ~UpdaterGPUScBFM_AB_Type();

    void initialize( int iGpuToUse );
    inline bool execute(){ return true; }
    void cleanup();

    /* setter methods */
    void copyBondSet( int dx, int dy, int dz, bool bondForbidden );
    inline void setAttribute( uint32_t i, int32_t attribute ){ mAttributeSystem[i] = attribute; }
    void setNrOfAllMonomers( uint32_t nAllMonomers );
    void setNetworkIngredients( uint32_t numPEG, uint32_t numPEGArm, uint32_t numCL );
    void setMonomerCoordinates( uint32_t i, int32_t x, int32_t y, int32_t z );
    void setConnectivity( uint32_t monoidx1, uint32_t monoidx2 );
    void setLatticeSize( uint32_t boxX, uint32_t boxY, uint32_t boxZ );

    /**
     * sets monomer positions given in mPolymerSystem in mLattice to occupied
     */
    void populateLattice();
    void runSimulationOnGPU( int32_t nrMCS_per_Call );

    int32_t getMonomerPositionInX( uint32_t i );
    int32_t getMonomerPositionInY( uint32_t i );
    int32_t getMonomerPositionInZ( uint32_t i );

    void setPeriodicity( bool isPeriodicX, bool isPeriodicY, bool isPeriodicZ );

};
