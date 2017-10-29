function checkSc()
{
    name='SimulatorCUDAGPUScBFM_AB_Type'
    sTime=$( date +%Y-%m-%dT%H-%M-%S )
    logFile=run-$name-$sTime.log
    rm result.bfm
    cmake -DCMAKE_BUILD_TYPE=Release -DINSTALLDIR_LEMONADE="$( cd .. && pwd )/install" .. &&
    make &&
    #time "./bin/$name" ../../test-files/Melt_N512_nc1024_LeMonADe_GPU_RKC_ScBFM.bfm 1000 1000 result.bfm |
    "./bin/$name" -i ../../test-files/Melt_N512_nc1024_LeMonADe_GPU_RKC_ScBFM.bfm -m 1000 -s 1000 -o result.bfm -e seeds.dat -g 1 |
    tee "run-$sTime.log"
    # cp result{,-norm}.bfm
    if [ -f result.bfm ]; then
        colordiff <( hexdump -C result-norm.bfm ) <( hexdump -C result.bfm ) | head -20
    fi
}
