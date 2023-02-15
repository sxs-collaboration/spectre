// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <tuple>
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
template <typename LowOrderReconstructor, bool PositivityPreserving,
          bool Use9thOrder, bool Use7thOrder>
struct PositivityPreservingAdaptiveOrderReconstructor {
  using ReturnType = std::tuple<double, double, std::uint8_t>;
  SPECTRE_ALWAYS_INLINE static ReturnType pointwise(
      const double* const u, const int stride, const double four_to_the_alpha_5,
      // GCC9 complains that six_to_the_alpha_7 and eight_to_the_alpha_9
      // are unused because if-constexpr
      [[maybe_unused]] const double six_to_the_alpha_7,
      [[maybe_unused]] const double eight_to_the_alpha_9) {
    using std::get;
    if constexpr (Use9thOrder) {
      const auto unlimited_9 = UnlimitedReconstructor<8>::pointwise(u, stride);
      const ReturnType order_9_result{get<0>(unlimited_9), get<1>(unlimited_9),
                                      9};

      if (not PositivityPreserving or LIKELY(get<0>(order_9_result) > 0.0 and
                                             get<1>(order_9_result) > 0.0)) {
        const double order_9_norm_of_top_modal_coefficient = square(
            -1.593380762005595 * u[stride] +
            0.7966903810027975 * u[2 * stride] -
            0.22762582314365648 * u[3 * stride] +
            0.02845322789295706 * u[4 * stride] -
            1.593380762005595 * u[-stride] +
            0.7966903810027975 * u[-2 * stride] -
            0.22762582314365648 * u[-3 * stride] +
            0.02845322789295706 * u[-4 * stride] + 1.991725952506994 * u[0]);

        const double order_9_norm_of_polynomial =
            u[stride] * (25.393963433621668 * u[stride] -
                         31.738453392103736 * u[2 * stride] +
                         14.315575523531798 * u[3 * stride] -
                         5.422933317103013 * u[4 * stride] +
                         45.309550145164756 * u[-stride] -
                         25.682667845756164 * u[-2 * stride] +
                         10.394184200706238 * u[-3 * stride] -
                         3.5773996341558414 * u[-4 * stride] -
                         56.63693768145594 * u[0]) +
            u[2 * stride] * (10.664627625179254 * u[2 * stride] -
                             9.781510753231265 * u[3 * stride] +
                             3.783820939683476 * u[4 * stride] -
                             25.682667845756164 * u[-stride] +
                             13.59830711617153 * u[-2 * stride] -
                             5.064486634342602 * u[-3 * stride] +
                             1.5850428636128617 * u[-4 * stride] +
                             33.99576779042882 * u[0]) +
            u[3 * stride] * (2.5801312593878514 * u[3 * stride] -
                             1.812843724346584 * u[4 * stride] +
                             10.394184200706238 * u[-stride] -
                             5.064486634342602 * u[-2 * stride] +
                             1.6716163773782988 * u[-3 * stride] -
                             0.4380794296257583 * u[-4 * stride] -
                             14.626643302060115 * u[0]) +
            u[4 * stride] * (0.5249097623867759 * u[4 * stride] -
                             3.5773996341558414 * u[-stride] +
                             1.5850428636128617 * u[-2 * stride] -
                             0.4380794296257583 * u[-3 * stride] +
                             0.07624062080823268 * u[-4 * stride] +
                             5.336843456576288 * u[0]) +
            u[-stride] * (25.393963433621668 * u[-stride] -
                          31.738453392103736 * u[-2 * stride] +
                          14.315575523531798 * u[-3 * stride] -
                          5.422933317103013 * u[-4 * stride] -
                          56.63693768145594 * u[0]) +
            u[-2 * stride] * (10.664627625179254 * u[-2 * stride] -
                              9.781510753231265 * u[-3 * stride] +
                              3.783820939683476 * u[-4 * stride] +
                              33.99576779042882 * u[0]) +
            u[-3 * stride] * (2.5801312593878514 * u[-3 * stride] -
                              1.812843724346584 * u[-4 * stride] -
                              14.626643302060115 * u[0]) +
            u[-4 * stride] * (0.5249097623867759 * u[-4 * stride] +
                              5.336843456576288 * u[0]) +
            33.758463458609164 * square(u[0]);
        if (square(eight_to_the_alpha_9) *
                order_9_norm_of_top_modal_coefficient <=
            order_9_norm_of_polynomial) {
          return order_9_result;
        }
      }
    }

    if constexpr (Use7thOrder) {
      const auto unlimited_7 = UnlimitedReconstructor<6>::pointwise(u, stride);
      const ReturnType order_7_result{get<0>(unlimited_7), get<1>(unlimited_7),
                                      7};

      if (not PositivityPreserving or LIKELY(get<0>(order_7_result) > 0.0 and
                                             get<1>(order_7_result) > 0.0)) {
        const double order_7_norm_of_top_modal_coefficient =
            square(0.06936287633138594 * u[-3 * stride] -
                   0.4161772579883155 * u[-2 * stride] +
                   1.040443144970789 * u[-stride] -  //
                   1.3872575266277185 * u[0] +       //
                   1.040443144970789 * u[stride] -
                   0.4161772579883155 * u[2 * stride] +  //
                   0.06936287633138594 * u[3 * stride]);

        const double order_7_norm_of_polynomial =
            u[stride] * (3.93094886671763 * u[stride] -
                         4.4887583031366605 * u[2 * stride] +
                         2.126671427664419 * u[3 * stride] +
                         6.081742742499426 * u[-stride] -
                         3.1180508323787337 * u[-2 * stride] +
                         1.2660604719155235 * u[-3 * stride] -
                         8.108990323332568 * u[0]) +
            u[2 * stride] * (1.7504056695205172 * u[2 * stride] -
                             1.402086588589091 * u[3 * stride] -
                             3.1180508323787337 * u[-stride] +
                             1.384291080027286 * u[-2 * stride] -
                             0.46498946172145633 * u[-3 * stride] +
                             4.614303600090953 * u[0]) +
            u[3 * stride] * (0.5786954880513824 * u[3 * stride] +
                             1.2660604719155235 * u[-stride] -
                             0.46498946172145633 * u[-2 * stride] +
                             0.10352871936656591 * u[-3 * stride] -
                             2.0705743873313183 * u[0]) +
            u[-stride] * (3.93094886671763 * u[-stride] -
                          4.4887583031366605 * u[-2 * stride] +
                          2.126671427664419 * u[-3 * stride] -
                          8.108990323332568 * u[0]) +
            u[-2 * stride] * (1.7504056695205172 * u[-2 * stride] -
                              1.402086588589091 * u[-3 * stride] +
                              4.614303600090953 * u[0]) +
            u[-3 * stride] * (0.5786954880513824 * u[-3 * stride] -
                              2.0705743873313183 * u[0]) +
            5.203166203165525 * square(u[0]);
        if (square(six_to_the_alpha_7) *
                order_7_norm_of_top_modal_coefficient <=
            order_7_norm_of_polynomial) {
          return order_7_result;
        }
      }
    }
    const auto unlimited_5 = UnlimitedReconstructor<4>::pointwise(u, stride);
    const ReturnType order_5_result{get<0>(unlimited_5), get<1>(unlimited_5),
                                    5};
    if (not PositivityPreserving or
        LIKELY(get<0>(order_5_result) > 0.0 and get<1>(order_5_result) > 0.0)) {
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
      if (square(four_to_the_alpha_5) * order_5_norm_of_top_modal_coefficient <=
          order_5_norm_of_polynomial) {
        return order_5_result;
      }
    }
    // Drop to low-order reconstructor
    const auto low_order = LowOrderReconstructor::pointwise(u, stride);
    const ReturnType low_order_result{get<0>(low_order), get<1>(low_order), 2};
    if (not PositivityPreserving or LIKELY(get<0>(low_order_result) > 0.0 and
                                           get<1>(low_order_result) > 0.0)) {
      return low_order_result;
    }
    // 1st-order reconstruction to guarantee positivity
    return {u[0], u[0], 1};
  }

  SPECTRE_ALWAYS_INLINE static constexpr size_t stencil_width() {
    return Use9thOrder ? 9 : (Use7thOrder ? 7 : 5);
  }
};
}  // namespace detail

/// @{
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
 *
 * If `Use9thOrder` is `true` then first a ninth-order reconstruction is used,
 * followed by fifth-order. If `Use7thOrder` is `true` then seventh-order
 * reconstruction is used before fifth-order (but after ninth-order if
 * `Use9thOrder` is also `true`). This allows using the highest possible order
 * locally for reconstruction.
 *
 * Fifth order unlimited reconstruction is:
 *
 * \f{align}{
 *   \hat{u}_{i+1/2}=\frac{3}{128}u_{i-2} - \frac{5}{32}u_{i-1}
 *     + \frac{45}{64}u_{i} + \frac{15}{32}u_{i+1} - \frac{5}{128}u_{i+2}
 * \f}
 *
 * Seventh order unlimited reconstruction is:
 *
 * \f{align}{
 *    \hat{u}_{i+1/2}&=-\frac{5}{1024}u_{i-3} + \frac{21}{512}u_{i-2}
 *         - \frac{175}{1024}u_{i-1} + \frac{175}{256}u_{i} \\
 *         &+ \frac{525}{1024}u_{i+1} - \frac{35}{512}u_{i+2}
 *         + \frac{7}{1024}u_{i+3}
 * \f}
 *
 * Ninth order unlimited reconstruction is:
 *
 * \f{align}{
 *   \hat{u}_{i+1/2}&=\frac{35}{32768}u_{i-4} - \frac{45}{4096}u_{i-3}
 *         + \frac{441}{8291}u_{i-2} - \frac{735}{4096}u_{i-1} \\
 *         &+ \frac{11025}{16384}u_{i}
 *         + \frac{2205}{4096}u_{i+1} - \frac{735}{8192}u_{i+2}
 *         + \frac{63}{4096}u_{i+3} \\
 *         &- \frac{45}{32768}u_{i+4}
 * \f}
 *
 */
template <typename LowOrderReconstructor, bool PositivityPreserving,
          bool Use9thOrder, bool Use7thOrder, size_t Dim>
void positivity_preserving_adaptive_order(
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        reconstructed_upper_side_of_face_vars,
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        reconstructed_lower_side_of_face_vars,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Index<Dim>& volume_extents, const size_t number_of_variables,
    const double four_to_the_alpha_5, const double six_to_the_alpha_7,
    const double eight_to_the_alpha_9) {
  detail::reconstruct<detail::PositivityPreservingAdaptiveOrderReconstructor<
      LowOrderReconstructor, PositivityPreserving, Use9thOrder, Use7thOrder>>(
      reconstructed_upper_side_of_face_vars,
      reconstructed_lower_side_of_face_vars, volume_vars, ghost_cell_vars,
      volume_extents, number_of_variables, four_to_the_alpha_5,
      six_to_the_alpha_7, eight_to_the_alpha_9);
}

template <typename LowOrderReconstructor, bool PositivityPreserving,
          bool Use9thOrder, bool Use7thOrder, size_t Dim>
void positivity_preserving_adaptive_order(
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        reconstructed_upper_side_of_face_vars,
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        reconstructed_lower_side_of_face_vars,
    const gsl::not_null<
        std::optional<std::array<gsl::span<std::uint8_t>, Dim>>*>
        reconstruction_order,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Index<Dim>& volume_extents, const size_t number_of_variables,
    const double four_to_the_alpha_5, const double six_to_the_alpha_7,
    const double eight_to_the_alpha_9) {
  detail::reconstruct<detail::PositivityPreservingAdaptiveOrderReconstructor<
      LowOrderReconstructor, PositivityPreserving, Use9thOrder, Use7thOrder>>(
      reconstructed_upper_side_of_face_vars,
      reconstructed_lower_side_of_face_vars, reconstruction_order, volume_vars,
      ghost_cell_vars, volume_extents, number_of_variables, four_to_the_alpha_5,
      six_to_the_alpha_7, eight_to_the_alpha_9);
}
/// @}

namespace detail {
template <size_t Dim, bool ReturnReconstructionOrder>
using ppao_recons_type = tmpl::conditional_t<
    ReturnReconstructionOrder,
    void (*)(
        gsl::not_null<std::array<gsl::span<double>, Dim>*>,
        gsl::not_null<std::array<gsl::span<double>, Dim>*>,
        gsl::not_null<std::optional<std::array<gsl::span<std::uint8_t>, Dim>>*>,
        const gsl::span<const double>&,
        const DirectionMap<Dim, gsl::span<const double>>&, const Index<Dim>&,
        size_t, double, double, double),
    void (*)(gsl::not_null<std::array<gsl::span<double>, Dim>*>,
             gsl::not_null<std::array<gsl::span<double>, Dim>*>,
             const gsl::span<const double>&,
             const DirectionMap<Dim, gsl::span<const double>>&,
             const Index<Dim>&, size_t, double, double, double)>;
}

/*!
 * \brief Returns function pointers to the
 * `positivity_preserving_adaptive_order` function, lower neighbor
 * reconstruction, and upper neighbor reconstruction.
 */
template <size_t Dim, bool ReturnReconstructionOrder>
auto positivity_preserving_adaptive_order_function_pointers(
    bool positivity_preserving, bool use_9th_order, bool use_7th_order,
    FallbackReconstructorType fallback_recons)
    -> std::tuple<detail::ppao_recons_type<Dim, ReturnReconstructionOrder>,
                  void (*)(gsl::not_null<DataVector*>, const DataVector&,
                           const DataVector&, const Index<Dim>&,
                           const Index<Dim>&, const Direction<Dim>&,
                           const double&, const double&, const double&),
                  void (*)(gsl::not_null<DataVector*>, const DataVector&,
                           const DataVector&, const Index<Dim>&,
                           const Index<Dim>&, const Direction<Dim>&,
                           const double&, const double&, const double&)>;
}  // namespace fd::reconstruction
