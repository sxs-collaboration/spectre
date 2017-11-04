// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/GenerateInstantiations.hpp"

// This is an example: replace the code with what you want to debug
#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GEN_OP(op, dim)                            \
  template bool operator op(const Index<dim>& lhs, \
                            const Index<dim>& rhs) noexcept;
#define INSTANTIATION(r, data)         \
  template class Index<GET_DIM(data)>; \
  GEN_OP(==, GET_DIM(data))            \
  GEN_OP(!=, GET_DIM(data))

GENERATE_INSTANTIATIONS(INSTANTIATION, (0, 1, 2, 3))

#undef GET_DIM
#undef GEN_OP
#undef INSTANTIATION
