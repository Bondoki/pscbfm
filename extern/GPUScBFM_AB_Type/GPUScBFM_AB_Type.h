#ifndef GPUScBFM_AB_Type_H_
#define GPUScBFM_AB_Type_H_


#include "./UpdaterGPUScBFM_AB_Type.h"


template<class IngredientsType>
class GPUScBFM_AB_Type : public AbstractUpdater {

private:
	UpdaterGPUScBFM_AB_Type Updater_GPUScBFM_AB_Type;

	int countGPU; //!< The idx of GPU to use.

	//! Number of mcs to be executed per GPU-call
	uint32_t nsteps;

protected:
  IngredientsType& ingredients;
  IngredientsType& getIngredients() {return ingredients;}

  typename IngredientsType::molecules_type& molecules;

public:
  /**
   * @brief Standard Constructor: Initialize the Ingredients and specify the GPU.
   *
   * @param val A reference to the IngredientsType - mainly the system
   * @param steps Number of mcs to be executed per GPU-call
   * @param whichGPU Index/count of the GPU to use. Default: 0
   */
	GPUScBFM_AB_Type(IngredientsType& val,uint32_t steps, int whichGPU=0):ingredients(val),molecules(val.modifyMolecules()),nsteps(steps), countGPU(whichGPU){};

	virtual ~GPUScBFM_AB_Type(){};

	void initialize() {

		// set nr of all monomers
		Updater_GPUScBFM_AB_Type.setNrOfAllMonomers(ingredients.getMolecules().size());

		// set periodicity
		Updater_GPUScBFM_AB_Type.setPeriodicity(ingredients.isPeriodicX(), ingredients.isPeriodicY(), ingredients.isPeriodicZ());

		//set num of HEP, PEG, PEGArm, Crosslinks
		Updater_GPUScBFM_AB_Type.setNetworkIngredients(ingredients.getNrOfStars(), ingredients.getNrOfMonomersPerStarArm(), ingredients.getNrOfCrosslinker());

		// copy monomer positions of all monomers
		for(int i = 0; i < ingredients.getMolecules().size(); i++)
			Updater_GPUScBFM_AB_Type.setMonomerCoordinates(i, molecules[i].getX(), molecules[i].getY(), molecules[i].getZ());


		// copy attributes of all monomers
		for(int i = 0; i < ingredients.getMolecules().size(); i++)
			Updater_GPUScBFM_AB_Type.setAttribute(i, ingredients.getMolecules()[i].getAttributeTag());

		// copy connectivity of all monomers
		for(int idx = 0; idx < ingredients.getMolecules().size(); idx++)
			for(int jbond = 0; jbond < ingredients.getMolecules().getNumLinks(idx); jbond++)
			{
				Updater_GPUScBFM_AB_Type.setConnectivity(idx, ingredients.getMolecules().getNeighborIdx(idx, jbond));
			}

		//set lattice size
		Updater_GPUScBFM_AB_Type.setLatticeSize(ingredients.getBoxX(), ingredients.getBoxY(), ingredients.getBoxZ());

		//populate lattice
		Updater_GPUScBFM_AB_Type.populateLattice();

		 //false-allowed; true-forbidden
		std::cout << "copy bondset" << std::endl;
		for (int dx=-4; dx <= 4; dx++)
			for (int dy=-4; dy <= 4; dy++)
				for (int dz=-4; dz <= 4; dz++)
				{
					Updater_GPUScBFM_AB_Type.copyBondSet(dx, dy, dz, !ingredients.getBondset().isValid(VectorInt3(dx, dy, dz)));
				}

		std::cout << "initialize GPU" << std::endl;
		Updater_GPUScBFM_AB_Type.initialize(countGPU);

	}
	virtual bool execute() {

		time_t startTimer = time(NULL); //in seconds
		std::cout<<"MCS:"<<ingredients.getMolecules().getAge()<<std::endl;

		std::cout << "start simulation on GPU"  << std::endl;

		Updater_GPUScBFM_AB_Type.runSimulationOnGPU(nsteps);

		std::cout << "copy back monomers"  << std::endl;
		// copy back positions of all monomers
		for(int idx = 0; idx < ingredients.getMolecules().size(); idx++)
			molecules[idx].setAllCoordinates(Updater_GPUScBFM_AB_Type.getMonomerPositionInX(idx), Updater_GPUScBFM_AB_Type.getMonomerPositionInY(idx), Updater_GPUScBFM_AB_Type.getMonomerPositionInZ(idx));

		ingredients.modifyMolecules().setAge(ingredients.modifyMolecules().getAge()+nsteps);

		std::cout<<"mcs "<<ingredients.getMolecules().getAge() << " with " << (((1.0*nsteps)*ingredients.getMolecules().size())/(difftime(time(NULL), startTimer)) ) << " [attempted moves/s]" <<std::endl;
		std::cout<<"mcs "<<ingredients.getMolecules().getAge() << " passed time " << ((difftime(time(NULL), startTimer)) ) << " [s] with " << nsteps << " MCS "<<std::endl;


		return true;

	}
	void cleanup() {
		std::cout << "cleanup" << std::endl;
		Updater_GPUScBFM_AB_Type.cleanup();
	}
};

#endif

