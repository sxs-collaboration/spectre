// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"
#include "NumericalAlgorithms/FiniteDifference/Reconstruct.hpp"
#include "NumericalAlgorithms/FiniteDifference/Unlimited.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Direction;
template <size_t Dim, typename T>
class DirectionMap;
template <size_t Dim>
class Index;
/// \endcond

namespace fd::reconstruction {
namespace detail {
template <typename LowOrderReconstructor, bool PositivityPreserving>
struct PositivityPreservingAdaptiveOrderReconstructor {
  SPECTRE_ALWAYS_INLINE static std::array<double, 2> pointwise(
      const double* const u, const int stride,
      const double four_to_the_alpha_5) {
    const std::array order_5_result =
        UnlimitedReconstructor<4>::pointwise(u, stride);
    if (not PositivityPreserving or
        LIKELY(order_5_result[0] > 0.0 and order_5_result[1] > 0.0)) {
      // The Persson sensor is 4^alpha L2(\hat{u}) <= L2(u)
      const double order_5_norm_of_top_modal_coefficient =
          0.2222222222222222 * square(-1.4880952380952381 * u[stride] +
                                      0.37202380952380953 * u[2 * stride] -
                                      1.4880952380952381 * u[-stride] +
                                      0.37202380952380953 * u[-2 * stride] +
                                      2.232142857142857 * u[0]);

      // Potential optimization: Simple approximation to the integral since we
      // only really need about 1 digit of accuracy. This does eliminate aliases
      // and so, while reducing the number of FLOPs, might not be accurate
      // enough in the cases we're interested in.
      //
      // const double order_5_norm_of_polynomial =
      //     1.1935763888888888 * square(u[-2 * stride]) +
      //     0.4340277777777778 * square(u[-stride]) +
      //     1.7447916666666667 * square(u[0]) +
      //     0.4340277777777778 * square(u[stride]) +
      //     1.1935763888888888 * square(u[2 * stride]);
      const double order_5_norm_of_polynomial =
          (u[stride] * (1.179711612654321 * u[stride] -
                        0.963946414792769 * u[2 * stride] +
                        1.0904086750440918 * u[-stride] -
                        0.5030502507716049 * u[-2 * stride] -
                        1.6356130125661377 * u[0]) +
           u[2 * stride] *
               (0.6699388830329586 * u[2 * stride] -
                0.5030502507716049 * u[-stride] +
                0.154568572944224 * u[-2 * stride] + 0.927411437665344 * u[0]) +
           u[-stride] * (1.179711612654321 * u[-stride] -
                         0.963946414792769 * u[-2 * stride] -
                         1.6356130125661377 * u[0]) +
           u[-2 * stride] * (0.6699388830329586 * u[-2 * stride] +
                             0.927411437665344 * u[0]) +
           1.4061182415674602 * square(u[0]));
      if (four_to_the_alpha_5 * order_5_norm_of_top_modal_coefficient <=
          order_5_norm_of_polynomial) {
        return order_5_result;
      }
    }
    // Drop to low-order reconstructor
    const auto low_order_result = LowOrderReconstructor::pointwise(u, stride);
    if (not PositivityPreserving or
        LIKELY(low_order_result[0] > 0.0 and low_order_result[1] > 0.0)) {
      return low_order_result;
    }
    // 1st-order reconstruction to guarantee positivity
    return {{u[0], u[0]}};
  }

  SPECTRE_ALWAYS_INLINE static constexpr size_t stencil_width() { return 5; }
};
}  // namespace detail

/*!
 * \ingroup FiniteDifferenceGroup
 * \brief Performs positivity-preserving adaptive-order FD reconstruction.
 *
 * Performs a fifth-order unlimited reconstruction. If the reconstructed
 * values at the interfaces aren't positive (when `PositivityPreserving` is
 * `true`) or when the Persson TCI condition:
 *
 * \f{align}{
 *  4^\alpha \int_{x_{i-5/2}}^{x_{i+5/2}} \hat{u}^2(x) dx
 *   > \int_{x_{i-5/2}}^{x_{i+5/2}} u^2(x) dx
 * \f}
 *
 * is satisfied, where \f$\hat{u}\f$ is the polynomial with only the largest
 * modal coefficient non-zero, then the `LowOrderReconstructor` is used.
 *
 * If `PositivityPreserving` is `true` then if the low-order reconstructed
 * solution isn't positive we use first-order (constant in space)
 * reconstruction.
 */
template <typename LowOrderReconstructor, bool PositivityPreserving, size_t Dim>
void positivity_preserving_adaptive_order(
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        reconstructed_upper_side_of_face_vars,
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        reconstructed_lower_side_of_face_vars,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Index<Dim>& volume_extents, const size_t number_of_variables,
    const double four_to_the_alpha_5) {
  detail::reconstruct<detail::PositivityPreservingAdaptiveOrderReconstructor<
      LowOrderReconstructor, PositivityPreserving>>(
      reconstructed_upper_side_of_face_vars,
      reconstructed_lower_side_of_face_vars, volume_vars, ghost_cell_vars,
      volume_extents, number_of_variables, four_to_the_alpha_5);
}

/*!
 * \brief Returns function pointers to the
 * `positivity_preserving_adaptive_order` function, lower neighbor
 * reconstruction, and upper neighbor reconstruction.
 */
template <size_t Dim>
auto positivity_preserving_adaptive_order_function_pointers(
    bool positivity_preserving, FallbackReconstructorType fallback_recons)
    -> std::tuple<
        void (*)(gsl::not_null<std::array<gsl::span<double>, Dim>*>,
                 gsl::not_null<std::array<gsl::span<double>, Dim>*>,
                 const gsl::span<const double>&,
                 const DirectionMap<Dim, gsl::span<const double>>&,
                 const Index<Dim>&, size_t, double),
        void (*)(gsl::not_null<DataVector*>, const DataVector&,
                 const DataVector&, const Index<Dim>&, const Index<Dim>&,
                 const Direction<Dim>&, const double&),
        void (*)(gsl::not_null<DataVector*>, const DataVector&,
                 const DataVector&, const Index<Dim>&, const Index<Dim>&,
                 const Direction<Dim>&, const double&)>;
}  // namespace fd::reconstruction
