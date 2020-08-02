cmake_minimum_required (VERSION 3.12)
project(mmdocr LANGUAGES CXX CUDA)
find_package(CUDA 11.0 REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17")

find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h)
find_library(AVCODEC_LIBRARY avcodec)

find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h)
find_library(AVFORMAT_LIBRARY avformat)

find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h)
find_library(AVUTIL_LIBRARY avutil)

add_library(lodepng STATIC lodepng.cpp lodepng.h)
add_library(nvcodecs STATIC NvCodecs/NvDecoder/NvDecoder.cpp NvCodecs/NvDecoder/NvDecoder.h)
add_library(ops STATIC CompareTextRegions.cu CRAFT.cu ExtractChangedFrames.cu ExtractTextRegions.cu OCR.cu PreprocessNV12.cu RgbBytesToRgbFp32.cu ExtractContiguousTextRegions.cu)
add_library(cpp_ops STATIC CTCDecoder.cpp SubtitleGenerator.cc utf_utils.cc common.cc)

target_compile_options(ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61>)
target_link_libraries(ops ${CUDA_LIBRARIES})
target_include_directories(
  ops PRIVATE
  ${CUDA_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME} main.cc)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

target_link_libraries(${PROJECT_NAME} ${AVCODEC_LIBRARY} ${AVFORMAT_LIBRARY} ${AVUTIL_LIBRARY})
target_link_libraries(${PROJECT_NAME} lodepng)
target_link_libraries(${PROJECT_NAME} nvcodecs)
target_link_libraries(${PROJECT_NAME} ops)
target_link_libraries(${PROJECT_NAME} cpp_ops)

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
target_include_directories(
  ${PROJECT_NAME} PRIVATE
  ${CUDA_INCLUDE_DIRS}
)
target_include_directories(
  cpp_ops PRIVATE
  ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} cuda)
target_link_libraries(ops cuda)
target_link_libraries(cpp_ops cuda)

target_link_libraries(nvcodecs ${CUDA_LIBRARIES})
target_include_directories(
  nvcodecs PRIVATE
  ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(nvcodecs /usr/local/cuda-11.0/lib64/libnvcuvid.so)


target_include_directories(${PROJECT_NAME} PRIVATE /home/zyddnys/opencv-cpu/include/opencv4)
target_link_libraries(${PROJECT_NAME} /home/zyddnys/opencv-cpu/lib/libopencv_world.so)
target_include_directories(ops PRIVATE /home/zyddnys/opencv-cpu/include/opencv4)
target_link_libraries(ops /home/zyddnys/opencv-cpu/lib/libopencv_world.so)
target_include_directories(cpp_ops PRIVATE /home/zyddnys/opencv-cpu/include/opencv4)
target_link_libraries(cpp_ops /home/zyddnys/opencv-cpu/lib/libopencv_world.so)

target_link_libraries(${PROJECT_NAME} nvinfer nvparsers nvonnxparser)
