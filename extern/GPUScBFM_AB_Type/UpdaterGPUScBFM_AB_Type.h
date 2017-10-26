/*
 * UpdaterGPUScBFM_AB_Type.h
 *
 *  Created on: 27.07.2017
 *      Author: Ron Dockhorn
 */

#ifndef UpdaterGPUScBFM_AB_Type_H_
#define UpdaterGPUScBFM_AB_Type_H_


#include <LeMonADE/utility/RandomNumberGenerators.h>

#include <iostream>
#include <stdio.h>
#include <stdexcept>
#include <stdint.h>
#include <sstream>

#include <cuda.h>

#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error)); exit(1);}}

#define NUMBLOCKS 16
#define NUMTHREADS 256

/*
typedef uint32_t uintCUDA;
typedef int32_t intCUDA;
#define MASK5BITS 2147483616
*/

typedef uint16_t uintCUDA;
typedef int16_t intCUDA;
#define MASK5BITS 32736


#define MAX_CONNECTIVITY 4

// #define NONPERIODICITY

typedef struct {
//	uint32_t size;
	uint32_t bondsMonomerIdx[MAX_CONNECTIVITY];
} MonoInfo ;

class UpdaterGPUScBFM_AB_Type {

private:
	//! Random Number Generator (RNG)
	RandomNumberGenerators randomNumbers;

	bool NotAllowedBondArray[512];
	//int BondAsciiArray[512];

	//! Number of all monomers in the system.
	uint32_t NrOfAllMonomers;

	uint32_t NrOfStars; //number of Stars
	uint32_t NrOfMonomersPerStarArm; //number OfMonomersPerStarArm
	uint32_t NrOfCrosslinker; //number of Crosslinker


	uint8_t *Lattice;
	int32_t *PolymerSystem;
	int32_t *AttributeSystem;

	//! Holds connectivity information
	struct MonoNNIndex {
		uint32_t size;
				uint32_t bondsMonomerIdx[MAX_CONNECTIVITY];
			};

	MonoNNIndex **monosNNidx;

    uint32_t Box_X;
    uint32_t Box_Y;
    uint32_t Box_Z;

    uint32_t Box_XM1;
    uint32_t Box_YM1;
    uint32_t Box_ZM1;

    uint32_t Box_XPRO;
    uint32_t Box_PROXY;

    intCUDA *PolymerSystem_device;
    intCUDA *PolymerSystem_host;

	uint8_t *LatticeOut_host;
	uint8_t *LatticeOut_device;

	uint8_t *LatticeTmp_device;
	uint8_t *LatticeTmp_host;

	MonoInfo *MonoInfo_host, *MonoInfo_device;   // Pointer to host & device arrays of structure

	uint32_t *MonomersSpeziesIdx_A_host;
	uint32_t *MonomersSpeziesIdx_B_host;

	uint32_t *MonomersSpeziesIdx_A_device;
	uint32_t *MonomersSpeziesIdx_B_device;

	uint32_t numblocksSpecies_A;
	uint32_t numblocksSpecies_B;


	int IndexBondArray(int x, int y, int z) {
		return (x & 7) + ((y & 7) << 3) + ((z & 7) << 6);
	}

	void checkSystem();

public:
	UpdaterGPUScBFM_AB_Type(){};
	virtual ~UpdaterGPUScBFM_AB_Type();

	void initialize(int countGPU);
	virtual bool execute();
	void cleanup();


	void copyBondSet(int, int, int, bool);

	void setNrOfAllMonomers(uint32_t nrOfAllMonomers);

	void setNetworkIngredients(uint32_t numPEG, uint32_t numPEGArm, uint32_t numCL);

	void setMonomerCoordinates(uint32_t idx, int32_t xcoor, int32_t ycoor, int32_t zcoor);

	void setAttribute(uint32_t idx, int32_t attrib);

	void setConnectivity(uint32_t monoidx1, uint32_t monoidx2);

	void setLatticeSize(uint32_t boxX, uint32_t boxY, uint32_t boxZ);

	void populateLattice();

	void runSimulationOnGPU(int32_t nrMCS_per_Call);

	int32_t getMonomerPositionInX(uint32_t idx);
	int32_t getMonomerPositionInY(uint32_t idx);
	int32_t getMonomerPositionInZ(uint32_t idx);

	void setPeriodicity(bool isPeriodicX, bool isPeriodicY, bool isPeriodicZ);

};


#endif /* UpdaterGPUScBFM_AB_Type_H_ */
