add_executable(matvec main.cpp kernels.cu)
add_executable(matvec_tests test.cu kernels.cu)

target_include_directories(matvec PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
)

target_include_directories(matvec_tests PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
)

target_link_libraries(matvec_tests PRIVATE Catch2::Catch2WithMain)

set_target_properties(matvec matvec_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# Only the test target requires cross-translation-unit device linking.
set_target_properties(matvec_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    OUTPUT_NAME test
)

catch_discover_tests(matvec_tests)
