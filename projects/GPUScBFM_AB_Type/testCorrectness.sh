function checkSc()
{
    name='SimulatorCUDAGPUScBFM_AB_Type'
    sTime=$( date +%Y-%m-%dT%H-%M-%S )
    logFile=run-$name-$sTime.log
    rm result.bfm
    cmake -DINSTALLDIR_LEMONADE="$( cd .. && pwd )/install" .. &&
    make &&
    time "./bin/$name" ../../test-files/Melt_N512_nc1024_LeMonADe_GPU_RKC_ScBFM.bfm 1000 1000 result.bfm |
    tee "run-$sTime.log"
    # cp result{,-norm}.bfm
    colordiff <( hexdump -C result-norm.bfm ) <( hexdump -C result.bfm )
}
