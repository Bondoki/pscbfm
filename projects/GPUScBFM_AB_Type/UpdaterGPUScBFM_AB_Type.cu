/*
 * UpdaterGPUScBFM_AB_Type.cpp
 *
 *  Created on: 27.07.2017
 *      Author: Ron Dockhorn
 */


#include "UpdaterGPUScBFM_AB_Type.h"

__device__ __constant__ bool IsBondForbiddenTable_d [512]; //false-allowed; true-forbidden

__device__ __constant__ intCUDA DXTable_d [6]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
__device__ __constant__ intCUDA DYTable_d [6]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
__device__ __constant__ intCUDA DZTable_d [6]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z

__device__ __constant__ uint32_t NrOfMonomersSpeciesA_d;  // Nr of Monomer Species A
__device__ __constant__ uint32_t NrOfMonomersSpeciesB_d;  // Nr of Monomer Species B

__device__ __constant__ uint32_t LATTICE_X_d;  // Lattice size in X
__device__ __constant__ uint32_t LATTICE_Y_d;  // Lattice size in Y
__device__ __constant__ uint32_t LATTICE_Z_d;  // Lattice size in Z

__device__ __constant__ uint32_t LATTICE_XM1_d;  // Lattice size in X-1
__device__ __constant__ uint32_t LATTICE_YM1_d;  // Lattice size in Y-1
__device__ __constant__ uint32_t LATTICE_ZM1_d;  // Lattice size in Z-1

__device__ __constant__ uint32_t LATTICE_XPRO_d;  // Lattice shift in X
__device__ __constant__ uint32_t LATTICE_PROXY_d;  // Lattice shift in X*Y

texture<uint8_t, cudaTextureType1D, cudaReadModeElementType> texLatticeRefOut;
texture<uint8_t, cudaTextureType1D, cudaReadModeElementType> texLatticeTmpRef;

texture<intCUDA, cudaTextureType1D, cudaReadModeElementType> texPolymerAndMonomerIsEvenAndOnXRef;

texture<int32_t, cudaTextureType1D, cudaReadModeElementType> texMonomersSpezies_A_ThreadIdx;
texture<int32_t, cudaTextureType1D, cudaReadModeElementType> texMonomersSpezies_B_ThreadIdx;


__device__ uint32_t hash(uint32_t a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

__device__ uintCUDA IdxBondArray_d(intCUDA x, intCUDA  y, intCUDA z) {
	return ((x & 7) + ((y & 7) << 3) + ((z & 7) << 6));
}


__global__ void runSimulationScBFMCheckSpeziesA_gpu(intCUDA *PolymerSystem_d, uint8_t *LatticeTmp_d, MonoInfo *MonoInfo_d , const  uint32_t rn) {


	  int idxA=blockIdx.x*blockDim.x+threadIdx.x;


	  if(idxA < NrOfMonomersSpeciesA_d)
	  {
		 //select random monomer
		 const uint32_t randomMonomer=tex1Dfetch(texMonomersSpezies_A_ThreadIdx,idxA);

		 //select random direction
		 const uintCUDA random_int = hash(hash(idxA) ^ rn) % 6;


		 const intCUDA xPosMono= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer);
		 const intCUDA yPosMono= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+1);
		 const intCUDA zPosMono= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+2);
		 const intCUDA MonoProperty = tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+3);

		  //select random direction
		  //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
		 const intCUDA dx = DXTable_d[random_int];
		 const intCUDA dy = DYTable_d[random_int];
		 const intCUDA dz = DZTable_d[random_int];

		 //check for periodicity if necessary
#ifdef NONPERIODICITY

		 if((xPosMono + dx) < 0)
			 return;

		 if((xPosMono + dx) >= LATTICE_XM1_d)
			 return;

		 if((yPosMono + dy) < 0)
			 return;

		 if((yPosMono + dy) >= LATTICE_YM1_d)
			 return;

		 if((zPosMono + dz) < 0)
			 return;

		 if((zPosMono + dz) >= LATTICE_ZM1_d)
			 return;

#endif

		 const unsigned nextNeigborSize = ((MonoProperty&224)>>5);


		  for(unsigned u=0; u < nextNeigborSize; u++)
	      {
			  const intCUDA nN_X=tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*MonoInfo_d[randomMonomer].bondsMonomerIdx[u]  );
			  const intCUDA nN_Y=tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*MonoInfo_d[randomMonomer].bondsMonomerIdx[u]+1);
			  const intCUDA nN_Z=tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*MonoInfo_d[randomMonomer].bondsMonomerIdx[u]+2);

		 	//IsBondForbiddenTable_d [512]; //false-allowed; true-forbidden
		 	if( IsBondForbiddenTable_d[IdxBondArray_d(nN_X-xPosMono-dx, nN_Y-yPosMono-dy, nN_Z-zPosMono-dz)] )
		 		    		return;

		  }



		  //check the lattice
		  uint8_t test = 0;

		  const uint32_t xPosMonoDXDX = ((xPosMono + dx + dx)&LATTICE_XM1_d);
		  const uint32_t yPosMonoDYDY = ((yPosMono + dy + dy)&LATTICE_YM1_d);
		  const uint32_t zPosMonoDZDZ = ((zPosMono + dz + dz)&LATTICE_ZM1_d);

		  const uint32_t xPosMonoAbs = ((xPosMono          )&LATTICE_XM1_d);
		  const uint32_t xPosMonoPDX = ((xPosMono + 1      )&LATTICE_XM1_d);
		  const uint32_t xPosMonoMDX = ((xPosMono - 1      )&LATTICE_XM1_d);

		  const uint32_t yPosMonoAbs = ((yPosMono          )&LATTICE_YM1_d);
		  const uint32_t yPosMonoPDY = ((yPosMono + 1      )&LATTICE_YM1_d);
		  const uint32_t yPosMonoMDY = ((yPosMono - 1      )&LATTICE_YM1_d);

		  const uint32_t zPosMonoAbs = ((zPosMono          )&LATTICE_ZM1_d);
		  const uint32_t zPosMonoPDZ = ((zPosMono + 1      )&LATTICE_ZM1_d);
		  const uint32_t zPosMonoMDZ = ((zPosMono - 1      )&LATTICE_ZM1_d);

		  switch (random_int >> 1)
			  {
			  case 0: //-+x

			  	  test =  tex1Dfetch(texLatticeRefOut, xPosMonoDXDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
				  	  	  tex1Dfetch(texLatticeRefOut, xPosMonoDXDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
				  	  	  tex1Dfetch(texLatticeRefOut, xPosMonoDXDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
				  	  	  tex1Dfetch(texLatticeRefOut, xPosMonoDXDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) ) |
				  	  	  tex1Dfetch(texLatticeRefOut, xPosMonoDXDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
				  	  	  tex1Dfetch(texLatticeRefOut, xPosMonoDXDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
				  	  	  tex1Dfetch(texLatticeRefOut, xPosMonoDXDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
				  	  	  tex1Dfetch(texLatticeRefOut, xPosMonoDXDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) ) |
				  	  	  tex1Dfetch(texLatticeRefOut, xPosMonoDXDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) );

				  	  break;

			  case 1: //-+y

				  test =  tex1Dfetch(texLatticeRefOut, xPosMonoMDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
				  		  tex1Dfetch(texLatticeRefOut, xPosMonoAbs + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
				  		  tex1Dfetch(texLatticeRefOut, xPosMonoPDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |

				  		  tex1Dfetch(texLatticeRefOut, xPosMonoMDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
				  		  tex1Dfetch(texLatticeRefOut, xPosMonoAbs + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
				  		  tex1Dfetch(texLatticeRefOut, xPosMonoPDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |

				  		  tex1Dfetch(texLatticeRefOut, xPosMonoMDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) ) |
				  		  tex1Dfetch(texLatticeRefOut, xPosMonoAbs + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) ) |
				  		  tex1Dfetch(texLatticeRefOut, xPosMonoPDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) );

				  	  break;

			  case 2: //-+z

		  		  test =  tex1Dfetch(texLatticeRefOut, xPosMonoMDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
		  		  	  	  tex1Dfetch(texLatticeRefOut, xPosMonoAbs + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
		  		  	  	  tex1Dfetch(texLatticeRefOut, xPosMonoPDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |

		  		  	  	  tex1Dfetch(texLatticeRefOut, xPosMonoMDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
		  		  	  	  tex1Dfetch(texLatticeRefOut, xPosMonoAbs + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
		  		  	  	  tex1Dfetch(texLatticeRefOut, xPosMonoPDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |

		  		  	  	  tex1Dfetch(texLatticeRefOut, xPosMonoMDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
		  		  	  	  tex1Dfetch(texLatticeRefOut, xPosMonoAbs + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
		  				  tex1Dfetch(texLatticeRefOut, xPosMonoPDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) );

			  		  break;

			  }

			  if (test) return;


		  // everything fits -> perform the move - add the information
			  // possible move

		    PolymerSystem_d[4*randomMonomer+3] = MonoProperty | ((random_int<<2)+1);

		    LatticeTmp_d[((xPosMono + dx     )&LATTICE_XM1_d) + (((yPosMono + dy	 )&LATTICE_YM1_d) << LATTICE_XPRO_d) + (((zPosMono + dz) & LATTICE_ZM1_d) << LATTICE_PROXY_d)]=1;








	  }
	}


__global__ void runSimulationScBFMPerformSpeziesA_gpu(intCUDA *PolymerSystem_d, uint8_t *Lattice_d) {


	  int idxA=blockIdx.x*blockDim.x+threadIdx.x;


	  if(idxA < NrOfMonomersSpeciesA_d)
	  {
		  //select random monomer
		  const uint32_t randomMonomer=tex1Dfetch(texMonomersSpezies_A_ThreadIdx,idxA);


		  const intCUDA MonoProperty = tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+3);


	  if(((MonoProperty&1) != 0))	//possible move
	  {
		  const intCUDA xPosMono= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer);
		  const intCUDA yPosMono= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+1);
		  const intCUDA zPosMono= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+2);

		  //select random direction
		  const uintCUDA random_int = (MonoProperty&28)>>2;

		  //select random direction
		  //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z


		  const intCUDA dx = DXTable_d[random_int];
		  const intCUDA dy = DYTable_d[random_int];
		  const intCUDA dz = DZTable_d[random_int];

		 //check the lattice
		 uint8_t test = 0;

		 const uint32_t xPosMonoDXDX = ((xPosMono + dx + dx)&LATTICE_XM1_d);
		 const uint32_t yPosMonoDYDY = ((yPosMono + dy + dy)&LATTICE_YM1_d);
		 const uint32_t zPosMonoDZDZ = ((zPosMono + dz + dz)&LATTICE_ZM1_d);

		 const  uint32_t xPosMonoAbs = ((xPosMono          )&LATTICE_XM1_d);
		 const uint32_t xPosMonoPDX = ((xPosMono + 1      )&LATTICE_XM1_d);
		 const uint32_t xPosMonoMDX = ((xPosMono - 1      )&LATTICE_XM1_d);

		 const uint32_t yPosMonoAbs = ((yPosMono          )&LATTICE_YM1_d);
		 const uint32_t yPosMonoPDY = ((yPosMono + 1      )&LATTICE_YM1_d);
		 const uint32_t yPosMonoMDY = ((yPosMono - 1      )&LATTICE_YM1_d);

		 const uint32_t zPosMonoAbs = ((zPosMono          )&LATTICE_ZM1_d);
		 const uint32_t zPosMonoPDZ = ((zPosMono + 1      )&LATTICE_ZM1_d);
		 const uint32_t zPosMonoMDZ = ((zPosMono - 1      )&LATTICE_ZM1_d);

		  switch (random_int >> 1)
			  {
			  case 0: //-+x

			  	  test =  tex1Dfetch(texLatticeTmpRef, xPosMonoDXDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
				  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoDXDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
				  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoDXDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
				  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoDXDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) ) |
				  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoDXDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
				  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoDXDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
				  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoDXDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
				  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoDXDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) ) |
				  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoDXDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) );

				  	  break;

			  case 1: //-+y

				  test =  tex1Dfetch(texLatticeTmpRef, xPosMonoMDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
				  		  tex1Dfetch(texLatticeTmpRef, xPosMonoAbs + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
				  		  tex1Dfetch(texLatticeTmpRef, xPosMonoPDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |

				  		  tex1Dfetch(texLatticeTmpRef, xPosMonoMDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
				  		  tex1Dfetch(texLatticeTmpRef, xPosMonoAbs + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
				  		  tex1Dfetch(texLatticeTmpRef, xPosMonoPDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |

				  		  tex1Dfetch(texLatticeTmpRef, xPosMonoMDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) ) |
				  		  tex1Dfetch(texLatticeTmpRef, xPosMonoAbs + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) ) |
				  		  tex1Dfetch(texLatticeTmpRef, xPosMonoPDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) );

				  	  break;

			  case 2: //-+z

		  		  test =  tex1Dfetch(texLatticeTmpRef, xPosMonoMDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
		  		  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoAbs + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
		  		  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoPDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |

		  		  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoMDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
		  		  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoAbs + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
		  		  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoPDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |

		  		  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoMDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
		  		  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoAbs + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
		  				  tex1Dfetch(texLatticeTmpRef, xPosMonoPDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) );

			  		  break;

			  }

			  if (test) return;


		  // everything fits -> perform the move - add the information
		    //PolymerSystem_d[4*randomMonomer  ] = xPosMono +dx;
		    //PolymerSystem_d[4*randomMonomer+1] = yPosMono +dy;
		    //PolymerSystem_d[4*randomMonomer+2] = zPosMono +dz;
		    PolymerSystem_d[4*randomMonomer+3] = MonoProperty | 2; // indicating allowed move
		    Lattice_d[((xPosMono + dx     )&LATTICE_XM1_d) + (((yPosMono + dy	 )&LATTICE_YM1_d) << LATTICE_XPRO_d) + (((zPosMono + dz) & LATTICE_ZM1_d) << LATTICE_PROXY_d)]=1;


		    Lattice_d[xPosMonoAbs + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d)]=0;

	  }

	}
}

__global__ void runSimulationScBFMZeroArraySpeziesA_gpu(intCUDA *PolymerSystem_d, uint8_t *LatticeTmp_d) {


	  int idxA=blockIdx.x*blockDim.x+threadIdx.x;

	  if(idxA < NrOfMonomersSpeciesA_d)
	  {
	  	  //select random monomer
		  const uint32_t randomMonomer=tex1Dfetch(texMonomersSpezies_A_ThreadIdx,idxA);


		  const intCUDA MonoProperty = tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+3);

	  if(((MonoProperty&3) != 0))	//possible move
	  {
		  const intCUDA xPosMono= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer);
		  const intCUDA yPosMono= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+1);
		  const intCUDA zPosMono= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+2);

		  //select random direction
		  const uintCUDA random_int = (MonoProperty&28)>>2;

		  //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
		  const intCUDA dx = DXTable_d[random_int];
		  const intCUDA dy = DYTable_d[random_int];
		  const intCUDA dz = DZTable_d[random_int];


		  // possible move but not allowed
		  if(((MonoProperty&3) == 1))
		  {
			  LatticeTmp_d[((xPosMono + dx     )&LATTICE_XM1_d) + (((yPosMono + dy	 )&LATTICE_YM1_d) << LATTICE_XPRO_d) + (((zPosMono + dz) & LATTICE_ZM1_d) << LATTICE_PROXY_d)]=0;

			  PolymerSystem_d[4*randomMonomer+3] = MonoProperty & MASK5BITS; // delete the first 5 bits
		  }
		  else //allowed move with all circumstance
		  {
			  PolymerSystem_d[4*randomMonomer  ] = xPosMono +dx;
			  PolymerSystem_d[4*randomMonomer+1] = yPosMono +dy;
			  PolymerSystem_d[4*randomMonomer+2] = zPosMono +dz;
			  PolymerSystem_d[4*randomMonomer+3] = MonoProperty & MASK5BITS; // delete the first 5 bits

			  LatticeTmp_d[((xPosMono + dx     )&LATTICE_XM1_d) + (((yPosMono + dy	 )&LATTICE_YM1_d) << LATTICE_XPRO_d) + (((zPosMono + dz) & LATTICE_ZM1_d) << LATTICE_PROXY_d)]=0;

			  //LatticeTmp_d[((xPosMono      )&LATTICE_XM1_d) + (((yPosMono 	 )&LATTICE_YM1_d) << LATTICE_XPRO_d) + (((zPosMono ) & LATTICE_ZM1_d) << LATTICE_PROXY_d)]=0;

		  }
		  // everything fits -> perform the move - add the information

	  }

	  }
}


__global__ void runSimulationScBFMCheckSpeziesB_gpu(intCUDA *PolymerSystem_d, uint8_t *LatticeTmp_d, MonoInfo *MonoInfo_d , const  uint32_t rn) {


	  int idxB=blockIdx.x*blockDim.x+threadIdx.x;


	  if(idxB < NrOfMonomersSpeciesB_d)
	  {
		  //select random monomer
		  const uint32_t randomMonomer=tex1Dfetch(texMonomersSpezies_B_ThreadIdx,idxB);

		  //select random direction
		  const uintCUDA random_int = hash(hash(idxB) ^ rn) % 6;

		  const intCUDA xPosMono= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer);
		  const intCUDA yPosMono= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+1);
		  const intCUDA zPosMono= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+2);
		  const intCUDA MonoProperty = tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+3);

		  //select random direction
		  //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
		  const intCUDA dx = DXTable_d[random_int];
		  const intCUDA dy = DYTable_d[random_int];
		  const intCUDA dz = DZTable_d[random_int];
		  const unsigned nextNeigborSize = ((MonoProperty&224)>>5);

#ifdef NONPERIODICITY

		 if((xPosMono + dx) < 0)
			 return;

		 if((xPosMono + dx) >= LATTICE_XM1_d)
			 return;

		 if((yPosMono + dy) < 0)
			 return;

		 if((yPosMono + dy) >= LATTICE_YM1_d)
			 return;

		 if((zPosMono + dz) < 0)
			 return;

		 if((zPosMono + dz) >= LATTICE_ZM1_d)
			 return;

#endif

		 for(unsigned u=0; u < nextNeigborSize; u++)
		 {
			  intCUDA nN_X=tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*MonoInfo_d[randomMonomer].bondsMonomerIdx[u]  );
			  intCUDA nN_Y=tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*MonoInfo_d[randomMonomer].bondsMonomerIdx[u]+1);
			  intCUDA nN_Z=tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*MonoInfo_d[randomMonomer].bondsMonomerIdx[u]+2);

		 		    	//IsBondForbiddenTable_d [512]; //false-allowed; true-forbidden
		 		    	if( IsBondForbiddenTable_d[IdxBondArray_d(nN_X-xPosMono-dx, nN_Y-yPosMono-dy, nN_Z-zPosMono-dz)] )
		 		    		return;

		 }

		//check the lattice
		uint8_t test = 0;

		const uint32_t xPosMonoDXDX = ((xPosMono + dx + dx)&LATTICE_XM1_d);
		const uint32_t yPosMonoDYDY = ((yPosMono + dy + dy)&LATTICE_YM1_d);
		const uint32_t zPosMonoDZDZ = ((zPosMono + dz + dz)&LATTICE_ZM1_d);

		const uint32_t xPosMonoAbs = ((xPosMono          )&LATTICE_XM1_d);
		const uint32_t xPosMonoPDX = ((xPosMono + 1      )&LATTICE_XM1_d);
		const uint32_t xPosMonoMDX = ((xPosMono - 1      )&LATTICE_XM1_d);

		const uint32_t yPosMonoAbs = ((yPosMono          )&LATTICE_YM1_d);
		const uint32_t yPosMonoPDY = ((yPosMono + 1      )&LATTICE_YM1_d);
		const uint32_t yPosMonoMDY = ((yPosMono - 1      )&LATTICE_YM1_d);

		const uint32_t zPosMonoAbs = ((zPosMono          )&LATTICE_ZM1_d);
		const uint32_t zPosMonoPDZ = ((zPosMono + 1      )&LATTICE_ZM1_d);
		const uint32_t zPosMonoMDZ = ((zPosMono - 1      )&LATTICE_ZM1_d);

		switch (random_int >> 1)
		{
		 case 0: //-+x
			 	 test =  tex1Dfetch(texLatticeRefOut, xPosMonoDXDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
			 	 	 	 tex1Dfetch(texLatticeRefOut, xPosMonoDXDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
			 	 	 	 tex1Dfetch(texLatticeRefOut, xPosMonoDXDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
			 	 	 	 tex1Dfetch(texLatticeRefOut, xPosMonoDXDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) ) |
			 	 	 	 tex1Dfetch(texLatticeRefOut, xPosMonoDXDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
			 	 	 	 tex1Dfetch(texLatticeRefOut, xPosMonoDXDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
			 	 	 	 tex1Dfetch(texLatticeRefOut, xPosMonoDXDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
			 	 	 	 tex1Dfetch(texLatticeRefOut, xPosMonoDXDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) ) |
			 	 	 	 tex1Dfetch(texLatticeRefOut, xPosMonoDXDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) );
			  	  break;

		  case 1: //-+y

			  test =  tex1Dfetch(texLatticeRefOut, xPosMonoMDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
			  		  tex1Dfetch(texLatticeRefOut, xPosMonoAbs + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
			  		  tex1Dfetch(texLatticeRefOut, xPosMonoPDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |

			  		  tex1Dfetch(texLatticeRefOut, xPosMonoMDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
			  		  tex1Dfetch(texLatticeRefOut, xPosMonoAbs + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
			  		  tex1Dfetch(texLatticeRefOut, xPosMonoPDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |

			  		  tex1Dfetch(texLatticeRefOut, xPosMonoMDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) ) |
			  		  tex1Dfetch(texLatticeRefOut, xPosMonoAbs + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) ) |
			  		  tex1Dfetch(texLatticeRefOut, xPosMonoPDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) );

			  	  break;

		  case 2: //-+z

	  		  test =  tex1Dfetch(texLatticeRefOut, xPosMonoMDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
	  		  	  	  tex1Dfetch(texLatticeRefOut, xPosMonoAbs + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
	  		  	  	  tex1Dfetch(texLatticeRefOut, xPosMonoPDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |

	  		  	  	  tex1Dfetch(texLatticeRefOut, xPosMonoMDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
	  		  	  	  tex1Dfetch(texLatticeRefOut, xPosMonoAbs + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
	  		  	  	  tex1Dfetch(texLatticeRefOut, xPosMonoPDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |

	  		  	  	  tex1Dfetch(texLatticeRefOut, xPosMonoMDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
	  		  	  	  tex1Dfetch(texLatticeRefOut, xPosMonoAbs + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
	  				  tex1Dfetch(texLatticeRefOut, xPosMonoPDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) );

		  		  break;

		}

		if (test) return;


		  // everything fits -> perform the move - add the information
			  // possible move
		  PolymerSystem_d[4*randomMonomer+3] = MonoProperty | ((random_int<<2)+1);

		  LatticeTmp_d[((xPosMono + dx     )&LATTICE_XM1_d) + (((yPosMono + dy	 )&LATTICE_YM1_d) << LATTICE_XPRO_d) + (((zPosMono + dz) & LATTICE_ZM1_d) << LATTICE_PROXY_d)]=1;








	  }
	}


__global__ void runSimulationScBFMPerformSpeziesB_gpu(intCUDA *PolymerSystem_d, uint8_t *Lattice_d) {


	  int idxB=blockIdx.x*blockDim.x+threadIdx.x;


	  if(idxB < NrOfMonomersSpeciesB_d)
	  {
		  //select random monomer
		  const uint32_t randomMonomer=tex1Dfetch(texMonomersSpezies_B_ThreadIdx,idxB);

		  const intCUDA MonoProperty = tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+3);


	if(((MonoProperty&1) != 0))	//possible move
	  {
		const intCUDA xPosMono= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer);
		const intCUDA yPosMono= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+1);
		const intCUDA zPosMono= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+2);

		  //select random direction
		const uintCUDA random_int = (MonoProperty&28)>>2;

		 //select random direction
		 //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z

		const intCUDA dx = DXTable_d[random_int];
		const intCUDA dy = DYTable_d[random_int];
		const intCUDA dz = DZTable_d[random_int];

		//check the lattice
		uint8_t test = 0;

		const uint32_t xPosMonoDXDX = ((xPosMono + dx + dx)&LATTICE_XM1_d);
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

			  	  test =  tex1Dfetch(texLatticeTmpRef, xPosMonoDXDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
				  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoDXDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
				  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoDXDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
				  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoDXDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) ) |
				  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoDXDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
				  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoDXDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
				  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoDXDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
				  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoDXDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) ) |
				  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoDXDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) );

				  	  break;

			  case 1: //-+y

				  test =  tex1Dfetch(texLatticeTmpRef, xPosMonoMDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
				  		  tex1Dfetch(texLatticeTmpRef, xPosMonoAbs + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |
				  		  tex1Dfetch(texLatticeTmpRef, xPosMonoPDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoMDZ << LATTICE_PROXY_d) ) |

				  		  tex1Dfetch(texLatticeTmpRef, xPosMonoMDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
				  		  tex1Dfetch(texLatticeTmpRef, xPosMonoAbs + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |
				  		  tex1Dfetch(texLatticeTmpRef, xPosMonoPDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d) ) |

				  		  tex1Dfetch(texLatticeTmpRef, xPosMonoMDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) ) |
				  		  tex1Dfetch(texLatticeTmpRef, xPosMonoAbs + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) ) |
				  		  tex1Dfetch(texLatticeTmpRef, xPosMonoPDX + (yPosMonoDYDY << LATTICE_XPRO_d) + (zPosMonoPDZ << LATTICE_PROXY_d) );

				  	  break;

			  case 2: //-+z

		  		  test =  tex1Dfetch(texLatticeTmpRef, xPosMonoMDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
		  		  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoAbs + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
		  		  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoPDX + (yPosMonoMDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |

		  		  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoMDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
		  		  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoAbs + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
		  		  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoPDX + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |

		  		  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoMDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
		  		  	  	  tex1Dfetch(texLatticeTmpRef, xPosMonoAbs + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) ) |
		  				  tex1Dfetch(texLatticeTmpRef, xPosMonoPDX + (yPosMonoPDY << LATTICE_XPRO_d) + (zPosMonoDZDZ << LATTICE_PROXY_d) );

			  		  break;

			  }

		if (test) return;


		  // everything fits -> perform the move - add the information

		    //PolymerSystem_d[4*randomMonomer  ] = xPosMono +dx;
		    //PolymerSystem_d[4*randomMonomer+1] = yPosMono +dy;
		    //PolymerSystem_d[4*randomMonomer+2] = zPosMono +dz;
		    PolymerSystem_d[4*randomMonomer+3] = MonoProperty | 2; // indicating allowed move
		    Lattice_d[((xPosMono + dx     )&LATTICE_XM1_d) + (((yPosMono + dy	 )&LATTICE_YM1_d) << LATTICE_XPRO_d) + (((zPosMono + dz) & LATTICE_ZM1_d) << LATTICE_PROXY_d)]=1;


		    Lattice_d[xPosMonoAbs + (yPosMonoAbs << LATTICE_XPRO_d) + (zPosMonoAbs << LATTICE_PROXY_d)]=0;

	  }

	}
}

__global__ void runSimulationScBFMZeroArraySpeziesB_gpu(intCUDA *PolymerSystem_d, uint8_t *LatticeTmp_d) {


	  int idxB=blockIdx.x*blockDim.x+threadIdx.x;

	  if(idxB < NrOfMonomersSpeciesB_d)
	  {
	  	  //select random monomer
		  const uint32_t randomMonomer=tex1Dfetch(texMonomersSpezies_B_ThreadIdx,idxB);


		  const intCUDA MonoProperty = tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+3);

	  if(((MonoProperty&3) != 0))	//possible move
	  {
		  const intCUDA xPosMono= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer);
		  const intCUDA yPosMono= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+1);
		  const intCUDA zPosMono= tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+2);

		  //select random direction
		  const uintCUDA random_int = (tex1Dfetch(texPolymerAndMonomerIsEvenAndOnXRef,4*randomMonomer+3)&28)>>2;

		  //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
		  const intCUDA dx = DXTable_d[random_int];
		  const intCUDA dy = DYTable_d[random_int];
		  const intCUDA dz = DZTable_d[random_int];



		  if(((MonoProperty&3) == 1))
		  {
			  LatticeTmp_d[((xPosMono + dx     )&LATTICE_XM1_d) + (((yPosMono + dy	 )&LATTICE_YM1_d) << LATTICE_XPRO_d) + (((zPosMono + dz) & LATTICE_ZM1_d) << LATTICE_PROXY_d)]=0;

			  PolymerSystem_d[4*randomMonomer+3] = MonoProperty & MASK5BITS; // delete the first 5 bits

		  }
		  else
		  {
			  PolymerSystem_d[4*randomMonomer  ] = xPosMono +dx;
			  PolymerSystem_d[4*randomMonomer+1] = yPosMono +dy;
			  PolymerSystem_d[4*randomMonomer+2] = zPosMono +dz;
			  PolymerSystem_d[4*randomMonomer+3] = MonoProperty & MASK5BITS; // delete the first 5 bits

			  LatticeTmp_d[((xPosMono + dx     )&LATTICE_XM1_d) + (((yPosMono + dy	 )&LATTICE_YM1_d) << LATTICE_XPRO_d) + (((zPosMono + dz) & LATTICE_ZM1_d) << LATTICE_PROXY_d)]=0;

			  //LatticeTmp_d[((xPosMono      )&LATTICE_XM1_d) + (((yPosMono 	 )&LATTICE_YM1_d) << LATTICE_XPRO_d) + (((zPosMono ) & LATTICE_ZM1_d) << LATTICE_PROXY_d)]=0;

		  }
		  // everything fits -> perform the move - add the information
		  //  PolymerSystem_d[4*randomMonomer+3] = MonoProperty & MASK5BITS; // delete the first 5 bits


	  }

	  }
}





UpdaterGPUScBFM_AB_Type::~UpdaterGPUScBFM_AB_Type() {

	std::cout << "destructor" << std::endl;

	delete[] Lattice;
	delete[] PolymerSystem;
	delete[] AttributeSystem;

	for (int a = 0; a < NrOfAllMonomers; ++a) // we will delete all of them
	{
		delete(monosNNidx[a]);

	}

	delete(monosNNidx);

}

void UpdaterGPUScBFM_AB_Type::initialize(int idxGPU) {

	cudaDeviceProp  prop;

	int count;
	CUDA_CHECK( cudaGetDeviceCount( &count ) );

	for (int i=0; i< count; i++) {
		CUDA_CHECK( cudaGetDeviceProperties( &prop, i ) );
		printf( "   --- General Information for device %d ---\n", i );
		printf( "Name:  %s\n", prop.name );
		printf( "Compute capability:  %d.%d\n", prop.major, prop.minor );
		printf( "Clock rate:  %d\n", prop.clockRate );
		printf( "Device copy overlap:  " );
		if (prop.deviceOverlap)
			printf( "Enabled\n" );
		else
			printf( "Disabled\n");
		printf( "Kernel execution timeout :  " );
		if (prop.kernelExecTimeoutEnabled)
			printf( "Enabled\n" );
		else
			printf( "Disabled\n" );

		printf( "   --- Memory Information for device %d ---\n", i );
		printf( "Total global mem:  %ld\n", prop.totalGlobalMem );
		printf( "Total constant Mem:  %ld\n", prop.totalConstMem );
		printf( "Max mem pitch:  %ld\n", prop.memPitch );
		printf( "Texture Alignment:  %ld\n", prop.textureAlignment );

		printf( "   --- MP Information for device %d ---\n", i );
		printf( "Multiprocessor count:  %d\n",
				prop.multiProcessorCount );
		printf( "Shared mem per mp:  %ld\n", prop.sharedMemPerBlock );
		printf( "Registers per mp:  %d\n", prop.regsPerBlock );
		printf( "Threads in warp:  %d\n", prop.warpSize );
		printf( "Max threads per block:  %d\n", prop.maxThreadsPerBlock );
		printf( "Max thread dimensions:  (%d, %d, %d)\n",
				prop.maxThreadsDim[0], prop.maxThreadsDim[1],
				prop.maxThreadsDim[2] );
		printf( "Max grid dimensions:  (%d, %d, %d)\n",
				prop.maxGridSize[0], prop.maxGridSize[1],
				prop.maxGridSize[2] );
		printf( "\n" );
	}

	if(idxGPU >= count)
	{
		std::cout << "GPU with idx " << idxGPU << " not present. Exiting..." << std::endl;
		throw std::runtime_error("Can not find GPU or GPU not present. Exiting...");
	}
	CUDA_CHECK( cudaSetDevice(idxGPU));

	// Init RNG
	//r250.Initialize(time(NULL));
	// is already initialized

	// create the BondTable and copy to constant memory
	 //false-allowed; true-forbidden
	std::cout << "copy BondTable: " << std::endl;
	bool *tmp_IsBondForbiddenTable = (bool *) malloc(sizeof(bool)*512);

	uint counter=0;
	for(int i = 0; i < 512; i++)
	{
		tmp_IsBondForbiddenTable[i]=false;
		tmp_IsBondForbiddenTable[i]=NotAllowedBondArray[i];

		std::cout << "bond: " << i << "  " << tmp_IsBondForbiddenTable[i] << "  " << NotAllowedBondArray[i] << std::endl;

		if (tmp_IsBondForbiddenTable[i] == false)
			counter++;
	}
	std::cout << "used bond in simulation: " << counter << " / 108 " << std::endl;

	if(counter != 108)
	{
		throw std::runtime_error("wrong bond-set!!! Exiting... \n");
	}

	CUDA_CHECK(cudaMemcpyToSymbol(IsBondForbiddenTable_d, tmp_IsBondForbiddenTable, sizeof(bool)*512));

	free(tmp_IsBondForbiddenTable);

	//creating the displacement arrays
	std::cout << "copy DXYZTable: " << std::endl;
	intCUDA *tmp_DXTable = (intCUDA *) malloc(sizeof(intCUDA)*6);
	intCUDA *tmp_DYTable = (intCUDA *) malloc(sizeof(intCUDA)*6);
	intCUDA *tmp_DZTable = (intCUDA *) malloc(sizeof(intCUDA)*6);

	tmp_DXTable[0]=-1; tmp_DXTable[1]= 1; tmp_DXTable[2]= 0; tmp_DXTable[3]= 0; tmp_DXTable[4]= 0; tmp_DXTable[5]= 0;
	tmp_DYTable[0]= 0; tmp_DYTable[1]= 0; tmp_DYTable[2]=-1; tmp_DYTable[3]= 1; tmp_DYTable[4]= 0; tmp_DYTable[5]= 0;
	tmp_DZTable[0]= 0; tmp_DZTable[1]= 0; tmp_DZTable[2]= 0; tmp_DZTable[3]= 0; tmp_DZTable[4]=-1; tmp_DZTable[5]= 1;

	CUDA_CHECK(cudaMemcpyToSymbol(DXTable_d, tmp_DXTable, sizeof(intCUDA)*6));
	CUDA_CHECK(cudaMemcpyToSymbol(DYTable_d, tmp_DYTable, sizeof(intCUDA)*6));
	CUDA_CHECK(cudaMemcpyToSymbol(DZTable_d, tmp_DZTable, sizeof(intCUDA)*6));

	free(tmp_DXTable);
	free(tmp_DYTable);
	free(tmp_DZTable);

    /***************************creating look-up for species*****************************************/

	uint32_t NrOfMonomersSpezies_A_host = 0;
	uint32_t NrOfMonomersSpezies_B_host = 0;

	uint32_t *MonomersSpezies_host =(uint32_t *) malloc((NrOfAllMonomers)*sizeof(uint32_t));

	for (uint32_t i=0; i<NrOfAllMonomers; i++)
		{
				//monomer is odd or even

				if(AttributeSystem[i] == 1)
				{
					MonomersSpezies_host[i]=1;
					//PolymerSystem_host[4*i+3]=0;
					//NrOfMonomersSpezies_A_host++;
				}
				if(AttributeSystem[i] == 2)
				{
					MonomersSpezies_host[i]=2;

				}
				if(AttributeSystem[i] == 0)
				{
					throw std::runtime_error("wrong attributes!!! Exiting... \n");
				}
		}

/*
	// NrOfMonomersPerStarArm is an odd number
	uint32_t NStar = 4*NrOfMonomersPerStarArm+1;
	
	for (uint32_t st=0; st<NrOfStars; st++)
			{
				//Center
				MonomersSpezies_host[st*NStar     ]=1;

				//first arm
				for(uint32_t onarm=1; onarm <= NrOfMonomersPerStarArm; onarm++)
				{
					uint32_t tag = (onarm%2)+1;
					MonomersSpezies_host[st*NStar + onarm ]=tag;
				}

				//second arm
				for(uint32_t onarm=1; onarm <= NrOfMonomersPerStarArm; onarm++)
				{
					uint32_t tag = (onarm%2)+1;
					MonomersSpezies_host[st*NStar + NrOfMonomersPerStarArm + onarm ]=tag;
				}

				//third arm
				for(uint32_t onarm=1; onarm <= NrOfMonomersPerStarArm; onarm++)
				{
					uint32_t tag = (onarm%2)+1;
					MonomersSpezies_host[st*NStar + 2*NrOfMonomersPerStarArm + onarm ]=tag;
				}

				//quad arm
				for(uint32_t onarm=1; onarm <= NrOfMonomersPerStarArm; onarm++)
				{
					uint32_t tag = (onarm%2)+1;
					MonomersSpezies_host[st*NStar + 3*NrOfMonomersPerStarArm + onarm ]=tag;
				}

		 	}

		//uint32_t offset = (NStar*NrOfStars);
		//
		//for (uint32_t i=0; i<NrOfCrosslinker; i++)
		//{
		//	MonomersSpezies_host[offset + i]=1;
		//}
		
		//for olympic the additional monomers behave as cross-linker
		for (uint32_t i=(NStar*NrOfStars); i<NrOfAllMonomers; i++)
		{
			MonomersSpezies_host[i]=1;
		}
		
*/

	for (uint32_t i=0; i<NrOfAllMonomers; i++)
		{
				//monomer is odd or even

				if( MonomersSpezies_host[i]==1)
					NrOfMonomersSpezies_A_host++;

				if( MonomersSpezies_host[i]==2)
					NrOfMonomersSpezies_B_host++;
		}

	std::cout << "NrOfMonomersSpezies_A: " << NrOfMonomersSpezies_A_host << std::endl;
	std::cout << "NrOfMonomersSpezies_B: " << NrOfMonomersSpezies_B_host << std::endl;

	if((NrOfMonomersSpezies_A_host+NrOfMonomersSpezies_B_host) != NrOfAllMonomers)
	{
		throw std::runtime_error("Nr Of MonomerSpezies doesn´t met!!! Exiting... \n");
	}

	MonomersSpeziesIdx_A_host =(uint32_t *) malloc((NrOfMonomersSpezies_A_host)*sizeof(uint32_t));
	MonomersSpeziesIdx_B_host =(uint32_t *) malloc((NrOfMonomersSpezies_B_host)*sizeof(uint32_t));

	uint32_t NrOfMonomersSpezies_A_host_dummy = 0;
	uint32_t NrOfMonomersSpezies_B_host_dummy = 0;

	for (uint32_t i=0; i<NrOfAllMonomers; i++)
	{
			//monomer is odd or even

			if( MonomersSpezies_host[i]==1)
			//else
			{
				MonomersSpeziesIdx_A_host[NrOfMonomersSpezies_A_host_dummy]=i;
				//PolymerSystem_host[4*i+3]=0;
				NrOfMonomersSpezies_A_host_dummy++;
			}

			if( MonomersSpezies_host[i]==2)
			{
				MonomersSpeziesIdx_B_host[NrOfMonomersSpezies_B_host_dummy]=i;
				//PolymerSystem_host[4*i+3]=32;
				NrOfMonomersSpezies_B_host_dummy++;
			}
	}

	if((NrOfMonomersSpezies_A_host != NrOfMonomersSpezies_A_host_dummy))
		{
			throw std::runtime_error("Nr Of MonomerSpezies_A_host doesn´t met!!! Exiting... \n");
		}

	if((NrOfMonomersSpezies_B_host != NrOfMonomersSpezies_B_host_dummy))
		{
			throw std::runtime_error("Nr Of MonomerSpezies_B_host doesn´t met!!! Exiting... \n");
		}


	std::cout << "create Look-Up-Thread-Table with size A: " << (NrOfMonomersSpezies_A_host)*sizeof(uint32_t) << " bytes = " << ((NrOfMonomersSpezies_A_host)*sizeof(uint32_t)/1024.0) << " kB "<< std::endl;
	std::cout << "create Look-Up-Thread-Table with size B: " << (NrOfMonomersSpezies_B_host)*sizeof(uint32_t) << " bytes = " << ((NrOfMonomersSpezies_B_host)*sizeof(uint32_t)/1024.0) << " kB "<< std::endl;

	CUDA_CHECK(cudaMalloc((void **) &MonomersSpeziesIdx_A_device, (NrOfMonomersSpezies_A_host)*sizeof(uint32_t)));
	CUDA_CHECK(cudaMalloc((void **) &MonomersSpeziesIdx_B_device, (NrOfMonomersSpezies_B_host)*sizeof(uint32_t)));

	std::cout << "copy Look-Up-Thread-Table with to GPU"<< std::endl;
	CUDA_CHECK(cudaMemcpy(MonomersSpeziesIdx_A_device, MonomersSpeziesIdx_A_host, (NrOfMonomersSpezies_A_host)*sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(MonomersSpeziesIdx_B_device, MonomersSpeziesIdx_B_host, (NrOfMonomersSpezies_B_host)*sizeof(uint32_t), cudaMemcpyHostToDevice));

	numblocksSpecies_A = (NrOfMonomersSpezies_A_host-1)/NUMTHREADS+1;
	std::cout << "calculate numBlocks Spezies A using" << (numblocksSpecies_A*NUMTHREADS) << "  needed: " << (NrOfMonomersSpezies_A_host) <<  std::endl;

	numblocksSpecies_B = (NrOfMonomersSpezies_B_host-1)/NUMTHREADS+1;
	std::cout << "calcluate numBlocks Spezies B using" << (numblocksSpecies_B*NUMTHREADS) << "  needed: " << (NrOfMonomersSpezies_B_host) <<  std::endl;

	//make constant:
	CUDA_CHECK(cudaMemcpyToSymbol(NrOfMonomersSpeciesA_d, &NrOfMonomersSpezies_A_host, sizeof(uint32_t)));
	CUDA_CHECK(cudaMemcpyToSymbol(NrOfMonomersSpeciesB_d, &NrOfMonomersSpezies_B_host, sizeof(uint32_t)));

	/************************end: creating look-up for species*****************************************/

	/****************************copy monomer informations ********************************************/


	PolymerSystem_host =(intCUDA *) malloc((4*NrOfAllMonomers+1)*sizeof(intCUDA));

	std::cout << "try to allocate : " << ((4*NrOfAllMonomers+1)*sizeof(intCUDA)) << " bytes = " << ((4*NrOfAllMonomers+1)*sizeof(intCUDA)/(1024.0)) << " kB = " << ((4*NrOfAllMonomers+1)*sizeof(intCUDA)/(1024.0*1024.0)) << " MB coordinates on GPU " << std::endl;

	CUDA_CHECK(cudaMalloc((void **) &PolymerSystem_device, (4*NrOfAllMonomers+1)*sizeof(intCUDA)));


	for (uint32_t i=0; i<NrOfAllMonomers; i++)
	{
		PolymerSystem_host[4*i]=(intCUDA) PolymerSystem[3*i];
		PolymerSystem_host[4*i+1]=(intCUDA) PolymerSystem[3*i+1];
		PolymerSystem_host[4*i+2]=(intCUDA) PolymerSystem[3*i+2];
		PolymerSystem_host[4*i+3]=0;
	}

	// prepare and copy the connectivity matrix to GPU
	// the index on GPU starts at 0 and is one less than loaded

	int sizeMonoInfo = NrOfAllMonomers * sizeof(MonoInfo);

	std::cout << "size of strut MonoInfo: " << sizeof(MonoInfo) << " bytes = " << (sizeof(MonoInfo)/(1024.0)) <<  "kB for one monomer connectivity " << std::endl;

	std::cout << "try to allocate : " << (sizeMonoInfo) << " bytes = " << (sizeMonoInfo/(1024.0)) <<  "kB = " << (sizeMonoInfo/(1024.0*1024.0)) <<  "MB for connectivity matrix on GPU " << std::endl;


	MonoInfo_host=(MonoInfo*) calloc(NrOfAllMonomers,sizeof(MonoInfo));
	CUDA_CHECK(  cudaMalloc((void **) &MonoInfo_device, sizeMonoInfo));   // Allocate array of structure on device


	for (uint32_t i=0; i<NrOfAllMonomers; i++)
		{
			//MonoInfo_host[i].size = monosNNidx[i]->size;
			if((monosNNidx[i]->size) > 7)
			{
				std::cout << "this GPU-model allows max 7 next neighbors but size is " << (monosNNidx[i]->size) << ". Exiting..." << std::endl;
				throw std::runtime_error("Limit of connectivity on GPU reached!!! Exiting... \n");
			}

			PolymerSystem_host[4*i+3] |= ((intCUDA)(monosNNidx[i]->size)) << 5;
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


	LatticeOut_host = (uint8_t *) malloc( LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t));

	LatticeTmp_host = (uint8_t *) malloc( LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t));

	std::cout << "try to allocate : " << (LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t)) << " bytes = " << (LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t)/(1024.0*1024.0)) << " MB lattice on GPU " << std::endl;


	CUDA_CHECK(cudaMalloc((void **) &LatticeOut_device, LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t)));
	CUDA_CHECK(cudaMalloc((void **) &LatticeTmp_device, LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t)));


	//copy information from Host to GPU
	for(int i = 0; i < LATTICE_X*LATTICE_Y*LATTICE_Z; i++)
	{
		LatticeOut_host[i]=0;
		LatticeTmp_host[i]=0;

	}

	//fill the tmpLattice - should be zero everywhere
	CUDA_CHECK(cudaMemcpy(LatticeTmp_device, LatticeTmp_host, LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t), cudaMemcpyHostToDevice));
	//start z-curve
	/*
	for (int t = 0; t < NrOfAllMonomers; t++) {

			uint32_t xk = (PolymerSystem[3*t  ]&LATTICE_XM1);
			uint32_t yk = (PolymerSystem[3*t+1]&LATTICE_YM1);
			uint32_t zk = (PolymerSystem[3*t+2]&LATTICE_ZM1);

			uint32_t inter3 = interleave3(xk/2,yk/2,zk/2);

			LatticeOut_host[((PolymerSystem_host[4*t+3] & 1) << 23) +inter3] = 1;

	}
	*/
	//end- z-curve

	for (int t = 0; t < NrOfAllMonomers; t++) {

			uint32_t xk = (PolymerSystem[3*t  ]&LATTICE_XM1);
			uint32_t yk = (PolymerSystem[3*t+1]&LATTICE_YM1);
			uint32_t zk = (PolymerSystem[3*t+2]&LATTICE_ZM1);

			//uint32_t inter3 = interleave3(xk/2,yk/2,zk/2);

			LatticeOut_host[xk + (yk << LATTICE_XPRO) + (zk << LATTICE_PROXY)] = 1;

			}



	std::cout << "checking the  LatticeOut_host: " << std::endl;
	CUDA_CHECK(cudaMemcpy(LatticeOut_device, LatticeOut_host, LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t), cudaMemcpyHostToDevice));


	//fetch from device and check again
	for(int i = 0; i < LATTICE_X*LATTICE_Y*LATTICE_Z; i++)
			Lattice[i]=0;

	std::cout << "copy back LatticeOut_host: " << std::endl;
	CUDA_CHECK(cudaMemcpy(LatticeOut_host, LatticeOut_device, LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t), cudaMemcpyDeviceToHost));

	for(int i = 0; i < LATTICE_X*LATTICE_Y*LATTICE_Z; i++)
		Lattice[i]=LatticeOut_host[i];

	std::cout << "copy back LatticeTmp_host: " << std::endl;
	CUDA_CHECK(cudaMemcpy(LatticeTmp_host, LatticeTmp_device, LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t), cudaMemcpyDeviceToHost));

	int dummyTmpCounter=0;
	for (int x=0;x<LATTICE_X;x++)
		for (int y=0;y<LATTICE_Y;y++)
			for (int z=0;z<LATTICE_Z;z++)
					 {
						dummyTmpCounter += (LatticeTmp_host[x + (y << LATTICE_XPRO) + (z << LATTICE_PROXY)]==0)? 0 : 1;
					 }
	std::cout << "occupied latticeTmp sites: " << dummyTmpCounter << " of " << (0) << std::endl;

	if(dummyTmpCounter != 0)
		throw std::runtime_error("Lattice occupation is wrong!!! Exiting... \n");

	//start -z-order
	/*
	cout << "recalculate Lattice: " << endl;
	//fetch from device and check again
	for(int i = 0; i < LATTICE_X*LATTICE_Y*LATTICE_Z; i++)
	{
		if(LatticeOut_host[i]==1)
		{
			uint32_t dummyhost = i;
			uint32_t onX = (dummyhost / (1 <<23)); //0 on O, 1 on X
			uint32_t zl = 2*( deinterleave3_Z((dummyhost % (1 <<23)))) + onX;
			uint32_t yl = 2*( deinterleave3_Y((dummyhost % (1 <<23)))) + onX;
			uint32_t xl = 2*( deinterleave3_X((dummyhost % (1 <<23)))) + onX;


			//cout << "X: " << xl << "\tY: " << yl << "\tZ: " << zl<< endl;
			Lattice[xl + (yl << LATTICE_XPRO) + (zl << LATTICE_PROXY)] = 1;

		}

	}
	*/
	//end -z-order


	/*************************end: creating lattice****************************************************/


	/*************************copy monomer positions***************************************************/
	CUDA_CHECK(cudaMemcpy(PolymerSystem_device, PolymerSystem_host, (4*NrOfAllMonomers+1)*sizeof(intCUDA), cudaMemcpyHostToDevice));
	/*************************end: copy monomer positions**********************************************/

	/*************************bind textures on GPU ****************************************************/
	std::cout << "bind textures "  << std::endl;
	//bind texture reference with linear memory
	cudaBindTexture(0,texLatticeRefOut,LatticeOut_device,LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t));

	cudaBindTexture(0,texLatticeTmpRef,LatticeTmp_device,LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t));

	cudaBindTexture(0,texPolymerAndMonomerIsEvenAndOnXRef,PolymerSystem_device,(4*NrOfAllMonomers+1)*sizeof(intCUDA));


	cudaBindTexture(0,texMonomersSpezies_A_ThreadIdx,MonomersSpeziesIdx_A_device,(NrOfMonomersSpezies_A_host)*sizeof(uint32_t));
	cudaBindTexture(0,texMonomersSpezies_B_ThreadIdx,MonomersSpeziesIdx_B_device,(NrOfMonomersSpezies_B_host)*sizeof(uint32_t));

	/*************************end: bind textures on GPU ************************************************/

	/*************************last check of system GPU *************************************************/

	CUDA_CHECK(cudaMemcpy(PolymerSystem_host, PolymerSystem_device, (4*NrOfAllMonomers+1)*sizeof(intCUDA), cudaMemcpyDeviceToHost));

	for (uint32_t i=0; i<NrOfAllMonomers; i++)
	{
		PolymerSystem[3*i  ]=(int32_t) PolymerSystem_host[4*i  ];
		PolymerSystem[3*i+1]=(int32_t) PolymerSystem_host[4*i+1];
		PolymerSystem[3*i+2]=(int32_t) PolymerSystem_host[4*i+2];
	}

	std::cout << "check system before simulation: " << std::endl;

	checkSystem();



	std::cout << "check system before simulation: " << std::endl;

	checkSystem();


	/*************************end: last check of system GPU *********************************************/

}

void UpdaterGPUScBFM_AB_Type::setNrOfAllMonomers(uint32_t nrOfAllMonomers) {
		NrOfAllMonomers = nrOfAllMonomers;

		std::cout << "used monomers in simulation: " << NrOfAllMonomers << std::endl;

		AttributeSystem = new int32_t[NrOfAllMonomers];
		PolymerSystem = new int32_t[3*NrOfAllMonomers+1];

		//idx is reduced by one compared to the file
		monosNNidx = new MonoNNIndex*[NrOfAllMonomers];

		for (int a = 0; a < NrOfAllMonomers; ++a)
		{
			monosNNidx[a] = new MonoNNIndex();

			monosNNidx[a]->size=0;

			for(unsigned o=0; o < MAX_CONNECTIVITY; o++)
			{
				monosNNidx[a]->bondsMonomerIdx[o]=0;
			}
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

void UpdaterGPUScBFM_AB_Type::setNetworkIngredients(uint32_t numPEG, uint32_t numPEGArm, uint32_t numCL) {

	NrOfStars = numPEG; //number of Stars
	NrOfMonomersPerStarArm = numPEGArm; //number OfMonomersPerStarArm
	NrOfCrosslinker = numCL; //number of Crosslinker

	std::cout << "NumPEG on GPU: " << NrOfStars << std::endl;
	std::cout << "NumPEGArmlength on GPU: " << NrOfMonomersPerStarArm << std::endl;
	std::cout << "NumCrosslinker on GPU: " << NrOfCrosslinker << std::endl;

	//if (NrOfMonomersPerStarArm != 29)
		//throw std::runtime_error("NrOfMonomersPerStarArm should be 29!!! Exiting...\n");

	//if ((NrOfMonomersPerStarArm%2) != 1)
		//	throw std::runtime_error("NrOfMonomersPerStarArm should be an odd number!!! Exiting...\n");

}

void UpdaterGPUScBFM_AB_Type::setMonomerCoordinates(uint32_t idx, int32_t xcoor, int32_t ycoor, int32_t zcoor){

	PolymerSystem[3*idx  ]=xcoor;
	PolymerSystem[3*idx+1]=ycoor;
	PolymerSystem[3*idx+2]=zcoor;

	//std::cout << i << "\t x:" << PolymerSystem[3*i  ] << "\t y:" << PolymerSystem[3*i+1]<< "\t z:" << PolymerSystem[3*i+2]<< std::endl;
}

void UpdaterGPUScBFM_AB_Type::setAttribute(uint32_t idx, int32_t attrib){

	//idx starts at 0
	AttributeSystem[idx]=attrib;
}

void UpdaterGPUScBFM_AB_Type::copyBondSet(int dx, int dy, int dz, bool bondNotAllowed)
{
	 //false-allowed; true-forbidden
	NotAllowedBondArray[IndexBondArray(dx,dy,dz)] = bondNotAllowed;
}

void UpdaterGPUScBFM_AB_Type::setConnectivity(uint32_t monoidx1, uint32_t monoidx2){

		monosNNidx[monoidx1]->bondsMonomerIdx[monosNNidx[monoidx1]->size] = monoidx2;
		//monosNNidx[monoidx2]->bondsMonomerIdx[monosNNidx[monoidx2]->size] = monoidx1;

		monosNNidx[monoidx1]->size++;
		//monosNNidx[monoidx2]->size++;

		//if((monosNNidx[monoidx1]->size > MAX_CONNECTIVITY) || (monosNNidx[monoidx2]->size > MAX_CONNECTIVITY))
		if((monosNNidx[monoidx1]->size > MAX_CONNECTIVITY))
		{
			throw std::runtime_error("MAX_CONNECTIVITY  exceeded!!! Exiting...\n");
		}
}

void UpdaterGPUScBFM_AB_Type::setLatticeSize(uint32_t boxX, uint32_t boxY, uint32_t boxZ){

	Box_X = boxX;
	Box_Y = boxY;
	Box_Z = boxZ;

	Box_XM1 = boxX-1;
	Box_YM1 = boxY-1;
	Box_ZM1 = boxZ-1;

	// determine the shift values for first multiplication
	uint32_t resultshift = -1;
	uint32_t dummy = boxX;
	while (dummy != 0) {
		dummy >>= 1;
		resultshift++;
	}
	Box_XPRO=resultshift;

	// determine the shift values for first multiplication
	resultshift = -1;
	dummy = boxX*boxY;
	while (dummy != 0) {
		dummy >>= 1;
		resultshift++;
	}
	Box_PROXY=resultshift;

	std::cout << "use bit shift for boxX: (1 << "<< Box_XPRO << " ) = " << (1 << Box_XPRO) << " = " << (boxX) << std::endl;
	std::cout << "use bit shift for boxX*boxY: (1 << "<< Box_PROXY << " ) = " << (1 << Box_PROXY) << " = " << (boxX*boxY) << std::endl;

	// check if shift is correct
	if ( (boxX != (1 << Box_XPRO)) || ((boxX*boxY) != (1 << Box_PROXY)) )
	{
		throw  std::runtime_error("Could not determine value for bit shift. Sure your box size is a power of 2? Exiting...\n");
	}

	//init lattice
	Lattice = new uint8_t[Box_X*Box_Y*Box_Z];

	for(int i = 0; i < Box_X*Box_Y*Box_Z; i++)
		Lattice[i]=0;
}

void UpdaterGPUScBFM_AB_Type::populateLattice()
{
	//if(!GPUScBFM.StartSimulationGPU())
	for (int idx = 0; idx < NrOfAllMonomers; idx++) {
		Lattice[(PolymerSystem[3*idx  ]&Box_XM1) + ((PolymerSystem[3*idx+1] &Box_YM1) << Box_XPRO) + ((PolymerSystem[3*idx+2]&Box_ZM1) << Box_PROXY)] = 1;
	}
}

void UpdaterGPUScBFM_AB_Type::checkSystem()
{
	std::cout << "checkSystem" << std::endl;


	int counterLatticeStart = 0;

		//if(!GPUScBFM.StartSimulationGPU())
	for(int i = 0; i < Box_X*Box_Y*Box_Z; i++)
		Lattice[i]=0;

	for (int idxMono=0; idxMono < (NrOfAllMonomers); idxMono++)
	{
		int32_t xpos = PolymerSystem[3*idxMono    ];
		int32_t ypos = PolymerSystem[3*idxMono+1  ];
		int32_t zpos = PolymerSystem[3*idxMono+2  ];

		Lattice[((0 + xpos) & Box_XM1) + (((0 + ypos) & Box_YM1)<< Box_XPRO) + (((0 + zpos) & Box_ZM1)<< Box_PROXY)]=1;
		Lattice[((1 + xpos) & Box_XM1) + (((0 + ypos) & Box_YM1)<< Box_XPRO) + (((0 + zpos) & Box_ZM1)<< Box_PROXY)]=1;
		Lattice[((0 + xpos) & Box_XM1) + (((1 + ypos) & Box_YM1)<< Box_XPRO) + (((0 + zpos) & Box_ZM1)<< Box_PROXY)]=1;
		Lattice[((1 + xpos) & Box_XM1) + (((1 + ypos) & Box_YM1)<< Box_XPRO) + (((0 + zpos) & Box_ZM1)<< Box_PROXY)]=1;

		Lattice[((0 + xpos) & Box_XM1) + (((0 + ypos) & Box_YM1)<< Box_XPRO) + (((1 + zpos) & Box_ZM1)<< Box_PROXY)]=1;
		Lattice[((1 + xpos) & Box_XM1) + (((0 + ypos) & Box_YM1)<< Box_XPRO) + (((1 + zpos) & Box_ZM1)<< Box_PROXY)]=1;
		Lattice[((0 + xpos) & Box_XM1) + (((1 + ypos) & Box_YM1)<< Box_XPRO) + (((1 + zpos) & Box_ZM1)<< Box_PROXY)]=1;
		Lattice[((1 + xpos) & Box_XM1) + (((1 + ypos) & Box_YM1)<< Box_XPRO) + (((1 + zpos) & Box_ZM1)<< Box_PROXY)]=1;

	}

	for (int x=0;x<Box_X;x++)
			for (int y=0;y<Box_Y;y++)
				for (int z=0;z<Box_Z;z++)
				 {
					counterLatticeStart += (Lattice[x + (y << Box_XPRO) + (z << Box_PROXY)]==0)? 0 : 1;
				    //if (Lattice[x + (y << LATTICE_XPRO) + (z << LATTICE_PROXY)] != 0)
				    //cout << x << " " << y << " " << z << "\t" <<  Lattice[x + (y << LATTICE_XPRO) + (z << LATTICE_PROXY)]<< endl;

				 }
	//counterLatticeStart *=8;

    std::cout << "occupied lattice sites: " << counterLatticeStart << " of " << (NrOfAllMonomers*8) << std::endl;

    if(counterLatticeStart != (NrOfAllMonomers*8))
    	throw std::runtime_error("Lattice occupation is wrong!!! Exiting... \n");


    std::cout << "check bonds" << std::endl;

    for (int idxMono=0; idxMono < (NrOfAllMonomers); idxMono++)
    	for(unsigned idxNN=0; idxNN < monosNNidx[idxMono]->size; idxNN++)
    	{
    		 int32_t bond_x = (PolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]  ]-PolymerSystem[3*idxMono  ]);
    		 int32_t bond_y = (PolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+1]-PolymerSystem[3*idxMono+1]);
    		 int32_t bond_z = (PolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+2]-PolymerSystem[3*idxMono+2]);

    		 if((bond_x > 3) || (bond_x < -3))
    		 {
    			 std::cout << "Invalid XBond..."<< std::endl;
    			 std::cout << bond_x<< " " << bond_y<< " " << bond_z<< "  between mono: " <<(idxMono+1)<< " (pos "<< PolymerSystem[3*idxMono  ] <<","<<PolymerSystem[3*idxMono+1]<<","<<PolymerSystem[3*idxMono+2] <<") and " << (monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+1) << " (pos "<< PolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]  ] <<","<<PolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+1]<<","+PolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+2] <<")"<<std::endl;
    			 throw std::runtime_error("Invalid XBond!!! Exiting...\n");
    		 }

    		 if((bond_y > 3) || (bond_y < -3))
    		 {
    			 std::cout << "Invalid YBond..."<< std::endl;
    			 std::cout << bond_x<< " " << bond_y<< " " << bond_z<< "  between mono: " <<(idxMono+1)<< " (pos "<< PolymerSystem[3*idxMono  ] <<","<<PolymerSystem[3*idxMono+1]<<","<<PolymerSystem[3*idxMono+2] <<") and " << (monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+1) << " (pos "<< PolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]  ] <<","<<PolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+1]<<","+PolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+2] <<")"<<std::endl;
    		     throw std::runtime_error("Invalid YBond!!! Exiting...\n");

    		 }

    		 if((bond_z > 3) || (bond_z < -3))
    		 {
    			 std::cout << "Invalid ZBond..."<< std::endl;
    			 std::cout << bond_x<< " " << bond_y<< " " << bond_z<< "  between mono: " <<(idxMono+1)<< " (pos "<< PolymerSystem[3*idxMono  ] <<","<<PolymerSystem[3*idxMono+1]<<","<<PolymerSystem[3*idxMono+2] <<") and " << (monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+1) << " (pos "<< PolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]  ] <<","<<PolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+1]<<","+PolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+2] <<")"<<std::endl;
    		     throw std::runtime_error("Invalid ZBond!!! Exiting...\n");
    		 }


    		 //false--erlaubt; true-forbidden
    		if( NotAllowedBondArray[IndexBondArray(bond_x, bond_y, bond_z)] )
    		{
    			std::cout << "something wrong with bonds between monomer: " << monosNNidx[idxMono]->bondsMonomerIdx[idxNN]  << " and " << idxMono << std::endl;
    			std::cout << (monosNNidx[idxMono]->bondsMonomerIdx[idxNN]) << "\t x: " << (PolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]])   << "\t y:" << PolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+1]<< "\t z:" << PolymerSystem[3*monosNNidx[idxMono]->bondsMonomerIdx[idxNN]+2]<< std::endl;
    			std::cout << idxMono << "\t x:" << PolymerSystem[3*idxMono  ] << "\t y:" << PolymerSystem[3*idxMono+1]<< "\t z:" << PolymerSystem[3*idxMono  ]<< std::endl;

    			throw std::runtime_error("Bond is NOT allowed!!! Exiting...\n");
    		}

    	}


}

void UpdaterGPUScBFM_AB_Type::runSimulationOnGPU(int32_t nrMCS){

	//time information
	time_t startTimer = time(NULL); //in seconds
	time_t difference = 0;

	//run simulation
	for (int32_t timeS =1; timeS <= nrMCS;timeS++)
	{
		/******* OneMCS ******/
		for(uint32_t cou = 0; cou < 2; cou++)
		{

		switch(randomNumbers.r250_rand32()%2) {


			case 0:  // run Spezies_A monomers
					runSimulationScBFMCheckSpeziesA_gpu<<<numblocksSpecies_A,NUMTHREADS>>>(PolymerSystem_device, LatticeTmp_device, MonoInfo_device, randomNumbers.r250_rand32());
				    runSimulationScBFMPerformSpeziesA_gpu<<<numblocksSpecies_A,NUMTHREADS>>>(PolymerSystem_device, LatticeOut_device);
				    runSimulationScBFMZeroArraySpeziesA_gpu<<<numblocksSpecies_A,NUMTHREADS>>>(PolymerSystem_device, LatticeTmp_device);
				    break;

			case 1: // run Spezies_B monomers
					runSimulationScBFMCheckSpeziesB_gpu<<<numblocksSpecies_B,NUMTHREADS>>>(PolymerSystem_device, LatticeTmp_device, MonoInfo_device, randomNumbers.r250_rand32());
					runSimulationScBFMPerformSpeziesB_gpu<<<numblocksSpecies_B,NUMTHREADS>>>(PolymerSystem_device, LatticeOut_device);
					runSimulationScBFMZeroArraySpeziesB_gpu<<<numblocksSpecies_B,NUMTHREADS>>>(PolymerSystem_device, LatticeTmp_device);

					break;

			default: break;
		}

		}

		/*
		if ((timeS%saveTime==0))
		{
			//copy information from GPU to Host

			//check the tmpLattice
			cout << "copy back LatticeTmp_host: " << endl;
			CUDA_CHECK(cudaMemcpy(LatticeTmp_host, LatticeTmp_device, LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t), cudaMemcpyDeviceToHost));

			int dummyTmpCounter=0;
			for (int x=0;x<LATTICE_X;x++)
				for (int y=0;y<LATTICE_Y;y++)
					for (int z=0;z<LATTICE_Z;z++)
							 {
								dummyTmpCounter += (LatticeTmp_host[x + (y << LATTICE_XPRO) + (z << LATTICE_PROXY)]==0)? 0 : 1;
							 }
			cout << "occupied latticeTmp sites: " << dummyTmpCounter << " of " << (0) << endl;



			CUDA_CHECK(cudaMemcpy(LatticeOut_host, LatticeOut_device, LATTICE_X*LATTICE_Y*LATTICE_Z*sizeof(uint8_t), cudaMemcpyDeviceToHost));

			for(int i = 0; i < LATTICE_X*LATTICE_Y*LATTICE_Z; i++)
					Lattice[i]=0;

			for(int i = 0; i < LATTICE_X*LATTICE_Y*LATTICE_Z; i++)
				Lattice[i]=LatticeOut_host[i];

			if(dummyTmpCounter != 0)
					exit(-1);

			//start -z-order
			//
			//cout << "save -- recalculate Lattice: " << endl;
			//fetch from device and check again
			//	for(int i = 0; i < LATTICE_X*LATTICE_Y*LATTICE_Z; i++)
			//	{
			//		if(LatticeOut_host[i]==1)
			//		{
			//			uint32_t dummyhost = i;
			//			uint32_t onX = (dummyhost / (1 <<23)); //0 on O, 1 on X
			//			uint32_t zl = 2*( deinterleave3_Z((dummyhost % (1 <<23)))) + onX;
			//			uint32_t yl = 2*( deinterleave3_Y((dummyhost % (1 <<23)))) + onX;
			//			uint32_t xl = 2*( deinterleave3_X((dummyhost % (1 <<23)))) + onX;


						//cout << "X: " << xl << "\tY: " << yl << "\tZ: " << zl<< endl;
			//			Lattice[xl + (yl << LATTICE_XPRO) + (zl << LATTICE_PROXY)] = 1;
						//
			//		}
					//
			//	}
				//end -z-order
			//


			CUDA_CHECK(cudaMemcpy(PolymerSystem_host, PolymerSystem_device, (4*NrOfAllMonomers+1)*sizeof(intCUDA), cudaMemcpyDeviceToHost));

			for (uint32_t i=0; i<NrOfAllMonomers; i++)
			{
				PolymerSystem[3*i]=(int32_t) PolymerSystem_host[4*i];
				PolymerSystem[3*i+1]=(int32_t) PolymerSystem_host[4*i+1];
				PolymerSystem[3*i+2]=(int32_t) PolymerSystem_host[4*i+2];
				//cout << i << "  : " << PolymerSystem_host[4*i+3] << endl;
			}

			checkSystem();


		    SaveSystem(timeS);
		    cout << "actual time: " << timeS << endl;

		    difference = time(NULL) - startTimer;
		    cout << "mcs = " << (timeS+MCSTime)  << "  speed [performed monomer try and move/s] = MCS*N/t: " << (1.0 * timeS * ((1.0 * NrOfAllMonomers) / (1.0f * difference))) << "     runtime[s]:" << (1.0f * difference) << endl;


		}
		*/
	}

	//All MCS are done- copy back...


		//copy information from GPU to Host

		//check the tmpLattice
		std::cout << "copy back LatticeTmp_host: " << std::endl;
		CUDA_CHECK(cudaMemcpy(LatticeTmp_host, LatticeTmp_device, Box_X*Box_Y*Box_Z*sizeof(uint8_t), cudaMemcpyDeviceToHost));

		int dummyTmpCounter=0;
		for (int x=0;x<Box_X;x++)
			for (int y=0;y<Box_Y;y++)
				for (int z=0;z<Box_Z;z++)
						 {
							dummyTmpCounter += (LatticeTmp_host[x + (y << Box_XPRO) + (z << Box_PROXY)]==0)? 0 : 1;
						 }
		std::cout << "occupied latticeTmp sites: " << dummyTmpCounter << " of " << (0) << std::endl;



		CUDA_CHECK(cudaMemcpy(LatticeOut_host, LatticeOut_device, Box_X*Box_Y*Box_Z*sizeof(uint8_t), cudaMemcpyDeviceToHost));

		for(int i = 0; i < Box_X*Box_Y*Box_Z; i++)
				Lattice[i]=0;

		for(int i = 0; i < Box_X*Box_Y*Box_Z; i++)
			Lattice[i]=LatticeOut_host[i];

		if(dummyTmpCounter != 0)
			throw std::runtime_error("Lattice occupation is wrong!!! Exiting... \n");

		//start -z-order
		/*
		cout << "save -- recalculate Lattice: " << endl;
		//fetch from device and check again
			for(int i = 0; i < LATTICE_X*LATTICE_Y*LATTICE_Z; i++)
			{
				if(LatticeOut_host[i]==1)
				{
					uint32_t dummyhost = i;
					uint32_t onX = (dummyhost / (1 <<23)); //0 on O, 1 on X
					uint32_t zl = 2*( deinterleave3_Z((dummyhost % (1 <<23)))) + onX;
					uint32_t yl = 2*( deinterleave3_Y((dummyhost % (1 <<23)))) + onX;
					uint32_t xl = 2*( deinterleave3_X((dummyhost % (1 <<23)))) + onX;


					//cout << "X: " << xl << "\tY: " << yl << "\tZ: " << zl<< endl;
					Lattice[xl + (yl << LATTICE_XPRO) + (zl << LATTICE_PROXY)] = 1;

				}

			}
			//end -z-order
		*/


		CUDA_CHECK(cudaMemcpy(PolymerSystem_host, PolymerSystem_device, (4*NrOfAllMonomers+1)*sizeof(intCUDA), cudaMemcpyDeviceToHost));

		for (uint32_t i=0; i<NrOfAllMonomers; i++)
		{
			PolymerSystem[3*i]=(int32_t) PolymerSystem_host[4*i];
			PolymerSystem[3*i+1]=(int32_t) PolymerSystem_host[4*i+1];
			PolymerSystem[3*i+2]=(int32_t) PolymerSystem_host[4*i+2];
			//cout << i << "  : " << PolymerSystem_host[4*i+3] << endl;
		}

		checkSystem();

	    std::cout << "run time (GPU): " << nrMCS << std::endl;

	    difference = time(NULL) - startTimer;
	    std::cout << "mcs = " << (nrMCS)  << "  speed [performed monomer try and move/s] = MCS*N/t: " << (1.0 * nrMCS * ((1.0 * NrOfAllMonomers) / (1.0f * difference))) << "     runtime[s]:" << (1.0f * difference) << std::endl;

}

int32_t UpdaterGPUScBFM_AB_Type::getMonomerPositionInX(uint32_t idx){
	return PolymerSystem[3*idx];
}

int32_t UpdaterGPUScBFM_AB_Type::getMonomerPositionInY(uint32_t idx){
	return PolymerSystem[3*idx+1];
}

int32_t UpdaterGPUScBFM_AB_Type::getMonomerPositionInZ(uint32_t idx){
	return PolymerSystem[3*idx+2];
}

bool UpdaterGPUScBFM_AB_Type::execute() {
	return true;
}

void UpdaterGPUScBFM_AB_Type::cleanup() {




	//copy information from GPU to Host
	CUDA_CHECK(cudaMemcpy(Lattice, LatticeOut_device, Box_X*Box_Y*Box_Z*sizeof(uint8_t), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaMemcpy(PolymerSystem_host, PolymerSystem_device, (4*NrOfAllMonomers+1)*sizeof(intCUDA), cudaMemcpyDeviceToHost));

	for (uint32_t i=0; i<NrOfAllMonomers; i++)
	{
		PolymerSystem[3*i]=(int32_t) PolymerSystem_host[4*i];
		PolymerSystem[3*i+1]=(int32_t) PolymerSystem_host[4*i+1];
		PolymerSystem[3*i+2]=(int32_t) PolymerSystem_host[4*i+2];
	}


	checkSystem();

	int sizeMonoInfo = NrOfAllMonomers * sizeof(MonoInfo);
	// copy connectivity matrix back from device to host
	CUDA_CHECK( cudaMemcpy(MonoInfo_host, MonoInfo_device, sizeMonoInfo, cudaMemcpyDeviceToHost));

	for (uint32_t i=0; i<NrOfAllMonomers; i++)
			{

				//if(MonoInfo_host[i].size != monosNNidx[i]->size)
				if(((PolymerSystem_host[4*i+3]&224)>>5) != monosNNidx[i]->size)
				{
					std::cout << "connectivity error after simulation run" << std::endl;
					std::cout << "mono:" << i << " vs " << (i) << std::endl;
					//cout << "numElements:" << MonoInfo_host[i].size << " vs " << monosNNidx[i]->size << endl;
					std::cout << "numElements:" << ((PolymerSystem_host[4*i+3]&224)>>5) << " vs " << monosNNidx[i]->size << std::endl;

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
	cudaUnbindTexture(texLatticeRefOut);
	cudaUnbindTexture(texLatticeTmpRef);
	cudaUnbindTexture(texPolymerAndMonomerIsEvenAndOnXRef);
	cudaUnbindTexture(texMonomersSpezies_A_ThreadIdx);
	cudaUnbindTexture(texMonomersSpezies_B_ThreadIdx);

	//free memory on GPU
	cudaFree(LatticeOut_device);
	cudaFree(LatticeTmp_device);

	cudaFree(PolymerSystem_device);
	cudaFree(MonoInfo_device);

	cudaFree(MonomersSpeziesIdx_A_device);
	cudaFree(MonomersSpeziesIdx_B_device);

	//free memory on CPU
	free(PolymerSystem_host);
	free(MonoInfo_host);

	free(LatticeOut_host);
	free(LatticeTmp_host);

	free(MonomersSpeziesIdx_A_host);
	free(MonomersSpeziesIdx_B_host);

}


/*inline bool UpdaterGPUScBFM_AB_Type::execute() {

	return true;
}


inline void UpdaterGPUScBFM_AB_Type::cleanup() {
}
*/
