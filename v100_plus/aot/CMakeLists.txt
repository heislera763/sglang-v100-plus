cmake_minimum_required(VERSION 3.26)
project(sgl-kernel-v100 LANGUAGES CXX CUDA)
find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
get_filename_component(AOT "${CMAKE_CURRENT_SOURCE_DIR}/../../python/sglang/kernels/aot" ABSOLUTE)
include(FetchContent)
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()
FetchContent_Declare(repo-cutlass URL https://github.com/NVIDIA/cutlass/archive/57e3cfb47a2d9e0d46eb6335c3dc411498efa198.tar.gz URL_HASH SHA256=09237099a70f80bff1dc8bb80c843a674bb4fdcb46e43cc6993e711c5ca89bb5)
FetchContent_Populate(repo-cutlass)
FetchContent_Declare(repo-flashinfer URL https://github.com/flashinfer-ai/flashinfer/archive/bc29697ba20b7e6bdb728ded98f04788e16ee021.tar.gz URL_HASH SHA256=931dfd118f4b6de8c7d98702153c7c03840139170af21a07607693bd9749744d)
FetchContent_Populate(repo-flashinfer)
set(SOURCES ops.cc)
foreach(source IN ITEMS
    elementwise/activation.cu elementwise/concat_mla.cu
    elementwise/fused_add_rms_norm_kernel.cu elementwise/pos_enc.cu
    grammar/apply_token_bitmask_inplace_cuda.cu mamba/causal_conv1d.cu
    memory/weak_ref_tensor.cpp moe/moe_align_kernel.cu
    moe/moe_sum.cu moe/moe_sum_reduce.cu
    moe/moe_topk_softmax_kernels.cu moe/moe_topk_sigmoid_kernels.cu
    speculative/eagle_utils.cu speculative/packbit.cu
    speculative/speculative_sampling.cu)
  list(APPEND SOURCES "${AOT}/csrc/${source}")
endforeach()
list(APPEND SOURCES "${repo-flashinfer_SOURCE_DIR}/csrc/norm.cu" "${repo-flashinfer_SOURCE_DIR}/csrc/renorm.cu")
Python_add_library(common_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})
target_include_directories(common_ops PRIVATE "${AOT}/include" "${AOT}/csrc" "${AOT}/../jit/include"
  "${repo-cutlass_SOURCE_DIR}/include" "${repo-flashinfer_SOURCE_DIR}/include" "${repo-flashinfer_SOURCE_DIR}/csrc")
target_compile_definitions(common_ops PRIVATE FLASHINFER_ENABLE_F16)
set_source_files_properties("${AOT}/csrc/speculative/speculative_sampling.cu" PROPERTIES COMPILE_DEFINITIONS SGLANG_V100_AOT_FLASHINFER_LEGACY_SCAN)
target_compile_options(common_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;--expt-extended-lambda;--threads=1>)
target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES} CUDA::cuda_driver CUDA::cudart)
# Upstream's loader already supports a generic library at the package root.
install(TARGETS common_ops LIBRARY DESTINATION sgl_kernel)
