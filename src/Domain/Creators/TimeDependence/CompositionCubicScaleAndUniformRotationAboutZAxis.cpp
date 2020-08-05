// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependence/UniformRotationAboutZAxis.hpp"

#include <limits>

#include "Domain/Creators/TimeDependence/Composition.hpp"
#include "Domain/Creators/TimeDependence/Composition.tpp"
#include "Domain/Creators/TimeDependence/CubicScale.hpp"
#include "Domain/Creators/TimeDependence/UniformRotationAboutZAxis.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace domain {
namespace creators {
namespace time_dependence {
/// \cond

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                               \
  template class Composition<                                                \
      TimeDependenceCompositionTag<CubicScale<GET_DIM(data)>,                \
                                   std::numeric_limits<size_t>::max()>,      \
      TimeDependenceCompositionTag<UniformRotationAboutZAxis<GET_DIM(data)>, \
                                   std::numeric_limits<size_t>::max()>>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (2, 3))

#undef GET_DIM
#undef INSTANTIATION

/// \endcond
}  // namespace time_dependence
}  // namespace creators
}  // namespace domain
