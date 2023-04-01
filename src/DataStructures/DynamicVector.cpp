// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DynamicVector.hpp"

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace DynamicVector_detail {
// Avoid including the entire option parser in a low-level header.
template <typename T>
std::vector<T> parse_to_vector(const Options::Option& options) {
  return options.parse_as<std::vector<T>>();
}

#define TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                        \
  template std::vector<TYPE(data)> parse_to_vector( \
      const Options::Option& options);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double))

#undef INSTANTIATE
#undef TYPE
}  // namespace DynamicVector_detail
