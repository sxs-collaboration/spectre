# Distributed under the MIT License.
# See LICENSE.txt for details.

# Set up position independent code by default since this is required
# for our python libraries.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
