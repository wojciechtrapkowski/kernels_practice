cmake_minimum_required(VERSION 3.27)

set(PROJECT_NAME Common_Tutorial)

project(${PROJECT_NAME} LANGUAGES C CXX)

file(GLOB SOURCES "source/tutorial/*.cpp" "source/tutorial/*.c")
file(GLOB HEADERS "include/tutorial/*.hpp" "include/tutorial/*.h")

add_library(${PROJECT_NAME} INTERFACE ${SOURCES} ${HEADERS})

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(${PROJECT_NAME} INTERFACE include)

target_link_libraries(${PROJECT_NAME} INTERFACE 
    OpenCL::OpenCL
)