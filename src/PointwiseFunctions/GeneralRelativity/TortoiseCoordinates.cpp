// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/TortoiseCoordinates.hpp"

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Simd/Simd.hpp"

namespace gr {

template <typename DataType>
DataType tortoise_radius_from_boyer_lindquist_minus_r_plus(
    const DataType& r_minus_r_plus, const double mass,
    const double dimensionless_spin) {
  const double r_plus = 1.0 + sqrt(1.0 - square(dimensionless_spin));
  return r_minus_r_plus + r_plus * mass +
         (r_plus * log(0.5 * r_minus_r_plus / mass) +
          (r_plus - 2.0) * log(0.5 * r_minus_r_plus / mass + r_plus - 1.0)) *
             mass / (r_plus - 1.0);
}

template <typename DataType>
DataType boyer_lindquist_radius_minus_r_plus_from_tortoise(
    const DataType& r_star, const double mass,
    const double dimensionless_spin) {
  const auto residual = [&r_star, &mass, &dimensionless_spin](
                            const auto r_minus_r_plus, const size_t i = 0) {
    if constexpr (simd::is_batch<
                      std::decay_t<decltype(r_minus_r_plus)>>::value) {
      return tortoise_radius_from_boyer_lindquist_minus_r_plus(
                 r_minus_r_plus, 1.0, dimensionless_spin) -
             simd::load_unaligned(&(get_element(r_star, i))) / mass;
    } else {
      return tortoise_radius_from_boyer_lindquist_minus_r_plus(
                 r_minus_r_plus, 1.0, dimensionless_spin) -
             get_element(r_star, i) / mass;
    }
  };
  // Possible performance optimization: tighten these bounds, e.g. by treating
  // small and large tortoise radii separately.
  const auto lower_bound = make_with_value<DataType>(r_star, 1e-14);
  DataType upper_bound = blaze::max(r_star / mass, 5.0);
  upper_bound =
      RootFinder::toms748(residual, lower_bound, upper_bound, 0.0, 1e-14)
          .value() *
      mass;
  return upper_bound;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                              \
  template DTYPE(data) tortoise_radius_from_boyer_lindquist_minus_r_plus( \
      const DTYPE(data) & r_minus_r_plus, double mass,                    \
      double dimensionless_spin);                                         \
  template DTYPE(data) boyer_lindquist_radius_minus_r_plus_from_tortoise( \
      const DTYPE(data) & r_star, double mass, double dimensionless_spin);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

}  // namespace gr
