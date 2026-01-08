project(Kernels)

add_library(Kernels INTERFACE)

file(GLOB KERNEL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cl")

add_custom_target(copy_kernels ALL)

foreach(KERNEL_FILE ${KERNEL_FILES})
    get_filename_component(KERNEL_NAME ${KERNEL_FILE} NAME_WE)
    string(TOUPPER ${KERNEL_NAME} KERNEL_NAME_UPPER)
    
    add_custom_command(TARGET copy_kernels POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory 
            ${CMAKE_BINARY_DIR}/$<CONFIG>/kernels
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${KERNEL_FILE}
            ${CMAKE_BINARY_DIR}/$<CONFIG>/kernels/${KERNEL_NAME}.cl
    )

    message(STATUS "Copying kernel: ${KERNEL_NAME}.cl")
    
    target_compile_definitions(Kernels INTERFACE 
        "${KERNEL_NAME_UPPER}_KERNEL_PATH=\"kernels/${KERNEL_NAME}.cl\""
    )
endforeach()