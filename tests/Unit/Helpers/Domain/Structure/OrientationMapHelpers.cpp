// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/Domain/Structure/OrientationMapHelpers.hpp"

#include <cstddef>
#include <vector>

#include "Domain/Structure/OrientationMap.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace TestHelpers::domain {
template <>
std::vector<OrientationMap<1>> valid_orientation_maps<1>() {
  return std::vector{OrientationMap<1>{},
                     OrientationMap<1>{std::array{Direction<1>::lower_xi()}}};
}

template <>
std::vector<OrientationMap<2>> valid_orientation_maps<2>() {
  return std::vector{OrientationMap<2>{},
                     OrientationMap<2>{std::array{Direction<2>::upper_eta(),
                                                  Direction<2>::lower_xi()}},
                     OrientationMap<2>{std::array{Direction<2>::lower_xi(),
                                                  Direction<2>::lower_eta()}},
                     OrientationMap<2>{std::array{Direction<2>::lower_eta(),
                                                  Direction<2>::upper_xi()}}};
}

template <>
std::vector<OrientationMap<3>> valid_orientation_maps<3>() {
  return std::vector{OrientationMap<3>{},
                     OrientationMap<3>{std::array{Direction<3>::lower_xi(),
                                                  Direction<3>::lower_eta(),
                                                  Direction<3>::upper_zeta()}},
                     OrientationMap<3>{std::array{Direction<3>::lower_xi(),
                                                  Direction<3>::upper_eta(),
                                                  Direction<3>::lower_zeta()}},
                     OrientationMap<3>{std::array{Direction<3>::upper_xi(),
                                                  Direction<3>::lower_eta(),
                                                  Direction<3>::lower_zeta()}},
                     OrientationMap<3>{std::array{Direction<3>::upper_eta(),
                                                  Direction<3>::upper_zeta(),
                                                  Direction<3>::upper_xi()}},
                     OrientationMap<3>{std::array{Direction<3>::lower_eta(),
                                                  Direction<3>::lower_zeta(),
                                                  Direction<3>::upper_xi()}},
                     OrientationMap<3>{std::array{Direction<3>::lower_eta(),
                                                  Direction<3>::upper_zeta(),
                                                  Direction<3>::lower_xi()}},
                     OrientationMap<3>{std::array{Direction<3>::upper_eta(),
                                                  Direction<3>::lower_zeta(),
                                                  Direction<3>::lower_xi()}},
                     OrientationMap<3>{std::array{Direction<3>::upper_zeta(),
                                                  Direction<3>::upper_xi(),
                                                  Direction<3>::upper_eta()}},
                     OrientationMap<3>{std::array{Direction<3>::lower_zeta(),
                                                  Direction<3>::lower_xi(),
                                                  Direction<3>::upper_eta()}},
                     OrientationMap<3>{std::array{Direction<3>::lower_zeta(),
                                                  Direction<3>::upper_xi(),
                                                  Direction<3>::lower_eta()}},
                     OrientationMap<3>{std::array{Direction<3>::upper_zeta(),
                                                  Direction<3>::lower_xi(),
                                                  Direction<3>::lower_eta()}},
                     OrientationMap<3>{std::array{Direction<3>::upper_eta(),
                                                  Direction<3>::lower_xi(),
                                                  Direction<3>::upper_zeta()}},
                     OrientationMap<3>{std::array{Direction<3>::lower_eta(),
                                                  Direction<3>::upper_xi(),
                                                  Direction<3>::upper_zeta()}},
                     OrientationMap<3>{std::array{Direction<3>::lower_eta(),
                                                  Direction<3>::lower_xi(),
                                                  Direction<3>::lower_zeta()}},
                     OrientationMap<3>{std::array{Direction<3>::upper_eta(),
                                                  Direction<3>::upper_xi(),
                                                  Direction<3>::lower_zeta()}},
                     OrientationMap<3>{std::array{Direction<3>::lower_xi(),
                                                  Direction<3>::upper_zeta(),
                                                  Direction<3>::upper_eta()}},
                     OrientationMap<3>{std::array{Direction<3>::upper_xi(),
                                                  Direction<3>::lower_zeta(),
                                                  Direction<3>::upper_eta()}},
                     OrientationMap<3>{std::array{Direction<3>::upper_xi(),
                                                  Direction<3>::upper_zeta(),
                                                  Direction<3>::lower_eta()}},
                     OrientationMap<3>{std::array{Direction<3>::lower_xi(),
                                                  Direction<3>::lower_zeta(),
                                                  Direction<3>::lower_eta()}},
                     OrientationMap<3>{std::array{Direction<3>::upper_zeta(),
                                                  Direction<3>::upper_eta(),
                                                  Direction<3>::lower_xi()}},
                     OrientationMap<3>{std::array{Direction<3>::lower_zeta(),
                                                  Direction<3>::lower_eta(),
                                                  Direction<3>::lower_xi()}},
                     OrientationMap<3>{std::array{Direction<3>::lower_zeta(),
                                                  Direction<3>::upper_eta(),
                                                  Direction<3>::upper_xi()}},
                     OrientationMap<3>{std::array{Direction<3>::upper_zeta(),
                                                  Direction<3>::lower_eta(),
                                                  Direction<3>::upper_xi()}}};
}

template <size_t Dim>
std::vector<OrientationMap<Dim>> random_orientation_maps(
    const size_t number_of_samples, gsl::not_null<std::mt19937*> generator) {
  return random_sample(number_of_samples, valid_orientation_maps<Dim>(),
                       generator);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                               \
  template std::vector<OrientationMap<DIM(data)>> random_orientation_maps( \
      const size_t number_of_samples, gsl::not_null<std::mt19937*> generator);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
}  // namespace TestHelpers::domain
