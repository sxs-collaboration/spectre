// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/CompressedMatrix.hpp"

#include "Options/ParseOptions.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace CompressedMatrix_detail {
// Avoid including the entire option parser in a low-level header.
template <typename Type>
std::vector<std::vector<Type>> parse_to_vectors(
    const Options::Option& options) {
  return options.parse_as<std::vector<std::vector<Type>>>();
}

#define TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                      \
  template std::vector<std::vector<TYPE(data)>> parse_to_vectors( \
      const Options::Option& options);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double))

#undef INSTANTIATE
#undef TYPE
}  // namespace CompressedMatrix_detail
