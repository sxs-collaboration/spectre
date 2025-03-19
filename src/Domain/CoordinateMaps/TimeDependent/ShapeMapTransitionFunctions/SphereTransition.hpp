// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/ShapeMapTransitionFunction.hpp"

namespace domain::CoordinateMaps::ShapeMapTransitionFunctions {

/*!
 * \ingroup CoordMapsTimeDependentGroup
 * \brief A transition function that falls off as $G(r,\theta,\phi) =
 * \frac{f(r)}{r} = \frac{ar + b}{r}$.
 *
 * \details The coefficients $a$ and $b$ are chosen so that the function $f(r) =
 * ar + b$ falls off linearly from 1 at \p r_min to 0 at \p r_max. The
 * coefficients are
 *
 * \f{align}{
 * \label{eq:transition_func}
 * a &= \frac{-1}{r_{\text{max}} - r_{\text{min}}} \\
 * b &= \frac{r_{\text{max}}}{r_{\text{max}} - r_{\text{min}}} = -a
 * r_{\text{max}}
 * \f}
 *
 * If \p reverse is set to `true`, then the function falls off from 0 at
 * \p r_min to 1 at \p r_max. To do this, the coefficients are modified as
 * $a \rightarrow -a$ and $b \rightarrow 1-b$.
 *
 * The function can be called within \p r_min, but only if \p interior is `true`
 * and \p reverse is `false`. Within \p r_min,
 *
 * \begin{equation}
 * G(r,\theta,\phi) = \frac{r^2}{r_{\text{min}}^3}.
 * \end{equation}
 *
 * which is chosen to match Eq. $\ref{eq:transition_func}$ at $r_{\text{min}}$
 * and go to 0 at $r=0$.
 *
 * This function cannot be called beyond \p r_max.
 *
 * If \p interior is `false` and if the `operator()` or `gradient()` is called
 * with a point within \p r_min, an error will occur and
 * `original_radius_over_radius` will return `std::nullopt`.
 *
 * This is a special, simplified, case of the
 * `domain::CoordinateMaps::ShapeMapTransitionFunctions::Wedge` class where both
 * the inner and outer surface are spheres centered on the same center.
 */
class SphereTransition final : public ShapeMapTransitionFunction {
 public:
  explicit SphereTransition() = default;
  SphereTransition(double r_min, double r_max, bool reverse = false,
                   bool interior = false);

  double operator()(
      const std::array<double, 3>& source_coords,
      const std::optional<size_t>& one_over_radius_power) const override;
  DataVector operator()(
      const std::array<DataVector, 3>& source_coords,
      const std::optional<size_t>& one_over_radius_power) const override;

  std::optional<double> original_radius_over_radius(
      const std::array<double, 3>& target_coords,
      double radial_distortion) const override;

  std::array<double, 3> gradient(
      const std::array<double, 3>& source_coords) const override;
  std::array<DataVector, 3> gradient(
      const std::array<DataVector, 3>& source_coords) const override;

  WRAPPED_PUPable_decl_template(SphereTransition);
  explicit SphereTransition(CkMigrateMessage* msg);
  void pup(PUP::er& p) override;

  std::unique_ptr<ShapeMapTransitionFunction> get_clone() const override {
    return std::make_unique<SphereTransition>(*this);
  }

  bool operator==(const ShapeMapTransitionFunction& other) const override;
  bool operator!=(const ShapeMapTransitionFunction& other) const override;

 private:
  template <typename T>
  T call_impl(const std::array<T, 3>& source_coords,
              const std::optional<size_t>& one_over_radius_power) const;

  template <typename T>
  std::array<T, 3> gradient_impl(const std::array<T, 3>& source_coords) const;

  double r_min_{};
  double inverse_cube_r_min_{};
  double r_max_{};
  double a_{};
  double b_{};
  bool interior_{};
  static constexpr double eps_ = std::numeric_limits<double>::epsilon() * 100;
};
}  // namespace domain::CoordinateMaps::ShapeMapTransitionFunctions
