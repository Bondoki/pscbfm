cmake_minimum_required (VERSION 2.8)

FIND_PACKAGE(CUDA REQUIRED)

#INCLUDE(FindCUDA)
#set(CUDA_INSTALL_PATH  /usr/local/cuda)
#set(CXX /opt/gcc-4.4.6/bin/g++)
#set(INCLUDED  ${CUDA_INSTALL_PATH}/include)

# sauron
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
#        -lineinfo -gencode arch=compute_20,code=sm_20 --compiler-bindir ${CXX} -I${INCLUDED})

# samweis
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lineinfo -gencode arch=compute_50,code=sm_50)

#for taurus
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lineinfo -gencode arch=compute_35,code=sm_35)

#for gandalf GTX1070
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lineinfo -gencode arch=compute_61,code=sm_61)

#for gandalf P100
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lineinfo -gencode arch=compute_60,code=sm_60)


set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --compiler-bindir /usr/bin/g++-4.9 )
set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lineinfo -gencode arch=compute_30,code=sm_30 )


if (NOT DEFINED LEMONADE_INCLUDE_DIR)
    message("LEMONADE_INCLUDE_DIR is not provided. If build fails, use -DLEMONADE_INCLUDE_DIR=/path/to/LeMonADE/headers/ or install to default location")
endif()

if (NOT DEFINED LEMONADE_LIBRARY_DIR)
    message("LEMONADE_LIBRARY_DIR is not provided. If build fails, use -DLEMONADE_LIBRARY_DIR=/path/to/LeMonADE/lib/ or install to default location")
endif()

# only use externally found include dirs, if this CMake file is not included as part of the LeMonADE compilation, else the dependency won't work, i.e. this file won't be recompiled if libLeMonADE changes
if( NOT APPLICATION_NAME STREQUAL "LeMonADE" )
    include_directories( ${LEMONADE_INCLUDE_DIR} )
endif()

add_library( mainCUDASimulatorGPUScBFM_AB_Type mainCUDASimulatorGPUScBFM_AB_Type.cpp )

cuda_add_executable( SimulatorCUDAGPUScBFM_AB_Type UpdaterGPUScBFM_AB_Type.cu )
#message("project_build_static = ${project_build_static}")
if( ( NOT DEFINED project_build_static ) OR ( project_build_static ) )
    target_link_libraries( SimulatorCUDAGPUScBFM_AB_Type mainCUDASimulatorGPUScBFM_AB_Type libLeMonADE-static )
else()
    link_directories ("${LEMONADE_LIBRARY_DIR}")
    target_link_libraries( SimulatorCUDAGPUScBFM_AB_Type mainCUDASimulatorGPUScBFM_AB_Type LeMonADE )
endif()



