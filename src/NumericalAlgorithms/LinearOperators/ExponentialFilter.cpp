// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/ExponentialFilter.hpp"

#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/StaticCache.hpp"

namespace Filters {

template <size_t FilterIndex>
Exponential<FilterIndex>::Exponential(const double alpha,
                                      const unsigned half_power,
                                      const bool disable_for_debugging)
    : alpha_(alpha),
      half_power_(half_power),
      disable_for_debugging_(disable_for_debugging) {}

template <size_t FilterIndex>
const Matrix& Exponential<FilterIndex>::filter_matrix(
    const Mesh<1>& mesh) const {
  const static double cached_alpha = alpha_;

  ASSERT(cached_alpha == alpha_, "Filter was cached with alpha = "
                                     << cached_alpha << ", but alpha is now "
                                     << alpha_
                                     << ".\nUse a different FilterIndex if you "
                                        "need a filter with new parameters\n");
  const static double cached_half_power = half_power_;
  ASSERT(cached_half_power == half_power_,
         "Filter was cached with half power = "
             << cached_half_power << ", but half power is now " << half_power_
             << ".\nUse a different FilterIndex if you need a filter with new "
                "parameters\n");

  const static auto cache = make_static_cache<
      CacheRange<1_st,
                 Spectral::maximum_number_of_points<Spectral::Basis::Legendre> +
                     1>,
      CacheEnumeration<Spectral::Basis, Spectral::Basis::Legendre,
                       Spectral::Basis::Chebyshev>,
      CacheEnumeration<Spectral::Quadrature, Spectral::Quadrature::Gauss,
                       Spectral::Quadrature::GaussLobatto>>(
      [alpha = alpha_, half_power = half_power_](
          const size_t extents, const Spectral::Basis basis,
          const Spectral::Quadrature quadrature) {
        return Spectral::filtering::exponential_filter(
            Mesh<1>{extents, basis, quadrature}, alpha, half_power);
      });
  return cache(mesh.extents(0), mesh.basis(0), mesh.quadrature(0));
}

template <size_t FilterIndex>
void Exponential<FilterIndex>::pup(PUP::er& p) {
  p | alpha_;
  p | half_power_;
  p | disable_for_debugging_;
}

template <size_t LocalFilterIndex>
bool operator==(const Exponential<LocalFilterIndex>& lhs,
                const Exponential<LocalFilterIndex>& rhs) {
  return lhs.alpha_ == rhs.alpha_ and lhs.half_power_ == rhs.half_power_ and
         lhs.disable_for_debugging_ == rhs.disable_for_debugging_;
}

template <size_t FilterIndex>
bool operator!=(const Exponential<FilterIndex>& lhs,
                const Exponential<FilterIndex>& rhs) {
  return not(lhs == rhs);
}

#define FILTER_INDEX(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GEN_OP(op, filter_index)                                  \
  template bool operator op(const Exponential<filter_index>& lhs, \
                            const Exponential<filter_index>& rhs);
#define INSTANTIATE(_, data)                      \
  template class Exponential<FILTER_INDEX(data)>; \
  GEN_OP(==, FILTER_INDEX(data))                  \
  GEN_OP(!=, FILTER_INDEX(data))

GENERATE_INSTANTIATIONS(INSTANTIATE, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))

#undef FILTER_INDEX
#undef INSTANTIATE

}  // namespace Filters
