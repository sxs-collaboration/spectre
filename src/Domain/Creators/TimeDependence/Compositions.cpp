// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependence/UniformRotationAboutZAxis.hpp"

#include "Domain/Creators/TimeDependence/Composition.hpp"
#include "Domain/Creators/TimeDependence/Composition.tpp"
#include "Domain/Creators/TimeDependence/CubicScale.hpp"
#include "Domain/Creators/TimeDependence/UniformRotationAboutZAxis.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace domain::creators::time_dependence {

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                 \
  template class Composition<                                  \
      TimeDependenceCompositionTag<CubicScale<GET_DIM(data)>>, \
      TimeDependenceCompositionTag<UniformRotationAboutZAxis<GET_DIM(data)>>>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (2, 3))

#undef INSTANTIATION

#define INSTANTIATION(r, data)                                         \
  template class Composition<                                          \
      TimeDependenceCompositionTag<UniformTranslation<GET_DIM(data)>>, \
      TimeDependenceCompositionTag<UniformTranslation<GET_DIM(data), 1>>>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef GET_DIM
}  // namespace domain::creators::time_dependence
