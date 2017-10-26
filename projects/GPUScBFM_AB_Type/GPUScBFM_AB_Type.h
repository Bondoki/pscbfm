#pragma once


#include <ctime>                        // clock
#include <iostream>

#include "./UpdaterGPUScBFM_AB_Type.h"

#define __FILENAME__ (__builtin_strrchr(__FILE__, '/') ? __builtin_strrchr(__FILE__, '/') + 1 : __FILE__)


/**
 * Why is this abstraction layer being used, instead of just incorporating
 * the GPU updated into this class ???
 */


template< class T_IngredientsType >
class GPUScBFM_AB_Type : public AbstractUpdater
{

private:
    UpdaterGPUScBFM_AB_Type mUpdaterGpu;

    int miGpuToUse;

    //! Number of Monte-Carlo Steps (mcs) to be executed (per GPU-call / Updater call)
    uint32_t mnSteps;

protected:
    T_IngredientsType & mIngredients;
    inline T_IngredientsType & getIngredients() { return mIngredients; }

    typename T_IngredientsType::molecules_type & molecules;

public:
    /**
     * @brief Standard constructor: initialize the ingredients and specify the GPU.
     *
     * @param rIngredients  A reference to the IngredientsType - mainly the system
     * @param rnSteps       Number of mcs to be executed per GPU-call
     * @param riGpuToUse    ID of the GPU to use. Default: 0
     */
    inline GPUScBFM_AB_Type
    (
        T_IngredientsType & rIngredients,
        uint32_t            rnSteps     ,
        int                 riGpuToUse = 0
    )
    : mIngredients( rIngredients                   ),
      molecules   ( rIngredients.modifyMolecules() ),
      mnSteps     ( rnSteps                        ),
      miGpuToUse  ( riGpuToUse                     )
    {}

    /**
     * https://stackoverflow.com/questions/461203/when-to-use-virtual-destructors
     * I don't see a reason for using 'virtual' if this class will never be
     * derived from.
     */
    inline ~GPUScBFM_AB_Type(){}

    /**
     * Copies required data and parameters from mIngredients to mUpdaterGpu
     * and calls the mUpdaterGpu initializer
     */
    inline void initialize()
    {
        /* Forward needed parameters to the GPU updater */
        mUpdaterGpu.setNrOfAllMonomers( mIngredients.getMolecules().size() );
        mUpdaterGpu.setPeriodicity( mIngredients.isPeriodicX(),
                                    mIngredients.isPeriodicY(),
                                    mIngredients.isPeriodicZ() );

        /* set num of HEP, PEG, PEGArm, Crosslinks. These getter functions
         * are provided by the FeatureNetwork.h feature */
        mUpdaterGpu.setNetworkIngredients( mIngredients.getNrOfStars(),
                                           mIngredients.getNrOfMonomersPerStarArm(),
                                           mIngredients.getNrOfCrosslinker() );

        /* copy monomer positions, attributes and connectivity of all monomers */
        for ( size_t i = 0u; i < mIngredients.getMolecules().size(); ++i )
        {
            mUpdaterGpu.setMonomerCoordinates( i, molecules[i].getX(),
                                                  molecules[i].getY(),
                                                  molecules[i].getZ() );
        }
        for ( size_t i = 0u; i < mIngredients.getMolecules().size(); ++i )
            mUpdaterGpu.setAttribute( i, mIngredients.getMolecules()[i].getAttributeTag() );
        for ( size_t i = 0u; i < mIngredients.getMolecules().size(); ++i )
        for ( size_t iBond = 0; iBond < mIngredients.getMolecules().getNumLinks(i); ++iBond )
            mUpdaterGpu.setConnectivity( i, mIngredients.getMolecules().getNeighborIdx( i, iBond ) );

        mUpdaterGpu.setLatticeSize( mIngredients.getBoxX(),
                                    mIngredients.getBoxY(),
                                    mIngredients.getBoxZ() );
        mUpdaterGpu.populateLattice();

         // false-allowed; true-forbidden
        std::cout << "[" << __FILENAME__ << "] copy bondset" << std::endl;
        /* maximum of (expected!!!) bond length in one dimension. Should be
         * queryable or there should be a better way to copy the bond set. */
        int const maxBondLength = 4;
        for ( int dx = -maxBondLength; dx <= maxBondLength; ++dx )
        for ( int dy = -maxBondLength; dy <= maxBondLength; ++dy )
        for ( int dz = -maxBondLength; dz <= maxBondLength; ++dz )
        {
            /* !!! The negation is confusing, again there should be a better way to copy the bond set */
            mUpdaterGpu.copyBondSet( dx, dy, dz, ! mIngredients.getBondset().isValid( VectorInt3( dx, dy, dz ) ) );
        }

        std::cout << "[" << __FILENAME__ << "] initialize GPU updater" << std::endl;
        mUpdaterGpu.initialize( miGpuToUse );
    }

    /**
     * Was the 'virtual' really necessary ??? I don't think there will ever be
     * some class inheriting from this class...
     * https://en.wikipedia.org/wiki/Virtual_function
     */
    inline bool execute()
    {
        std::clock_t const t0 = std::clock();

        std::cout << "[" << __FILENAME__ << "] MCS:" << mIngredients.getMolecules().getAge() << std::endl;
        std::cout << "[" << __FILENAME__ << "] start simulation on GPU" << std::endl;

        mUpdaterGpu.runSimulationOnGPU( mnSteps );

        // copy back positions of all monomers
        std::cout << "[" << __FILENAME__ << "] copy back monomers from GPU updater to CPU 'molecules' to be used with analyzers" << std::endl;
        for( size_t i = 0; i < mIngredients.getMolecules().size(); ++i )
        {
            molecules[i].setAllCoordinates
            (
                mUpdaterGpu.getMonomerPositionInX(i),
                mUpdaterGpu.getMonomerPositionInY(i),
                mUpdaterGpu.getMonomerPositionInZ(i)
            );
        }

        /* update number of total simulation steps already done */
        mIngredients.modifyMolecules().setAge( mIngredients.modifyMolecules().getAge()+ mnSteps );

        std::clock_t const t1 = std::clock();
        double const dt = (double) ( t1 - t0 ) / CLOCKS_PER_SEC;    // in seconds
        /* attempted moves per second */
        double const amps = ( (double) mnSteps * mIngredients.getMolecules().size() )/ dt;

        std::cout
        << "[" << __FILENAME__ << "] mcs " << mIngredients.getMolecules().getAge()
        << " with " << amps << " [attempted moves/s]\n"
        << "[" << __FILENAME__ << "] mcs " << mIngredients.getMolecules().getAge()
        << " passed time " << dt << " [s] with " << mnSteps << " MCS "
        << std::endl;

        return true;
    }

    inline void cleanup()
    {
        std::cout << "[" << __FILENAME__ << "] cleanup" << std::endl;
        mUpdaterGpu.cleanup();
    }
};
