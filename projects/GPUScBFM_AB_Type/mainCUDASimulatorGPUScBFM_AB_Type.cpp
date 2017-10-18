
#include <cstring>

#include <LeMonADE/utility/RandomNumberGenerators.h>
#include <LeMonADE/core/ConfigureSystem.h>
#include <LeMonADE/core/Ingredients.h>
#include <LeMonADE/feature/FeatureMoleculesIO.h>
#include <LeMonADE/feature/FeatureAttributes.h>
#include <LeMonADE/feature/FeatureExcludedVolume.h>
#include <LeMonADE/feature/FeatureFixedMonomers.h>
#include <LeMonADE/utility/TaskManager.h>
#include <LeMonADE/updater/UpdaterReadBfmFile.h>
#include <LeMonADE/updater/UpdaterSimpleSimulator.h>


#include "./GPUScBFM_AB_Type.h"
#include "./FeatureNetwork.h"


int main(int argc, char* argv[])
{
  try{
	std::string infile;
	std::string outfile;
	uint32_t max_mcs=0;
	uint32_t save_interval=0;

	outfile="outfile.bfm";

	if(!(argc==4 || argc==5 )|| (argc==2 && strcmp(argv[1],"--help")==0 ))
	{
		std::string errormessage;
		errormessage="usage: ./SimulatorCUDAGPUScBFM_AB_Type input_filename max_mcs save_interval(mcs) [output_filename]\n";
		errormessage+="\nSimple Simulator for the ScBFM with Ex.Vol and BondCheck splitted CL-PEG in z on GPU\n";
		throw std::runtime_error(errormessage);

	}
	else
	{
		infile=argv[1];
		max_mcs=atoi(argv[2]);
		save_interval=atoi(argv[3]);

		if(argc==5) outfile=argv[4];
		else outfile=argv[1];
	}


	//seed the globally available random number generators
	RandomNumberGenerators rng;
	rng.seedAll();

	// FeatureExcludedVolume<> is equivalent to FeatureExcludedVolume<FeatureLattice<bool> >
	//typedef LOKI_TYPELIST_3(FeatureBondset,FeatureAttributes,FeatureLattice<uint8_t>FeatureExcludedVolume<FeatureLatticePowerOfTwo<> >) Features;
	typedef LOKI_TYPELIST_4(FeatureMoleculesIO,  FeatureAttributes, FeatureExcludedVolume<>, FeatureNetwork) Features;

	typedef ConfigureSystem<VectorInt3,Features, 4> Config;
	typedef Ingredients<Config> Ing;
	Ing myIngredients;

	TaskManager taskmanager;
	taskmanager.addUpdater(new UpdaterReadBfmFile<Ing>(infile,myIngredients,UpdaterReadBfmFile<Ing>::READ_LAST_CONFIG_SAVE),0);
	//here you can choose to use MoveLocalBcc instead. Careful though: no real tests made yet
	//(other than for latticeOccupation, valid bonds, frozen monomers...)
	taskmanager.addUpdater(new GPUScBFM_AB_Type<Ing>(myIngredients,save_interval,0));

	taskmanager.addAnalyzer(new AnalyzerWriteBfmFile<Ing>(outfile,myIngredients));

	taskmanager.initialize();
	taskmanager.run(max_mcs/save_interval);
	taskmanager.cleanup();

	}
	catch(std::exception& err){std::cerr<<err.what();}
	return 0;

}

