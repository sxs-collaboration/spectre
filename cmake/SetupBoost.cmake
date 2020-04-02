# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Boost 1.60.0 REQUIRED COMPONENTS program_options)

message(STATUS "Boost include: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost libraries: ${Boost_LIBRARIES}")

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "Boost Version:  ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}\n"
  )

# Boost organizes targets as:
# - Boost::boost is the header-only parts of Boost
# - Boost::COMPONENT are the components that need linking, e.g. program_options

add_interface_lib_headers(
  TARGET Boost::boost
  HEADERS
  boost/config.hpp
  boost/core/demangle.hpp
  boost/functional/hash.hpp
  boost/integer/common_factor_rt.hpp
  boost/make_shared.hpp
  boost/multi_array.hpp
  boost/none.hpp
  boost/optional.hpp
  boost/parameter/name.hpp
  boost/preprocessor/arithmetic/inc.hpp
  boost/preprocessor/control/expr_iif.hpp
  boost/preprocessor/control/iif.hpp
  boost/preprocessor/control/while.hpp
  boost/preprocessor/list/adt.hpp
  boost/preprocessor/list/fold_left.hpp
  boost/preprocessor/list/fold_right.hpp
  boost/preprocessor/list/for_each_product.hpp
  boost/preprocessor/list/size.hpp
  boost/preprocessor/list/to_tuple.hpp
  boost/preprocessor/list/transform.hpp
  boost/preprocessor/logical/bitand.hpp
  boost/preprocessor/logical/bool.hpp
  boost/preprocessor/logical/compl.hpp
  boost/preprocessor/repetition/for.hpp
  boost/preprocessor/tuple/elem.hpp
  boost/preprocessor/tuple/reverse.hpp
  boost/preprocessor/tuple/size.hpp
  boost/preprocessor/tuple/to_list.hpp
  boost/preprocessor/variadic/elem.hpp
  boost/preprocessor/variadic/to_list.hpp
  boost/range/combine.hpp
  boost/shared_ptr.hpp
  boost/tuple/tuple.hpp
  boost/variant.hpp
  )

set_property(
  TARGET Boost::program_options
  APPEND PROPERTY PUBLIC_HEADER
  boost/program_options.hpp
  )
