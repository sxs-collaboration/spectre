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
 * \brief A transition function meant to be used in the
 * domain::CoordinateMaps::TimeDependent::Shape map.
 *
 * The formula for this transition function is
 *
 * \f{align}{
 * f(r) &= \left\{\begin{array}{ll} 1 , & r <= r_{\rm min}, \\
 *      \frac{ar + b}{r}, & r_{\rm min} < r < r_{\rm max}, \\
 *      0, & r_{\rm max} <= r,\end{array}\right.
 * \f}
 *
 * where the coefficients \f$a\f$ and \f$b\f$ are chosen so that the map falls
 * off linearly from 1 at `r_min` to 0 at `r_max`.
 */
class SphereTransition final : public ShapeMapTransitionFunction {
 public:
  explicit SphereTransition() = default;
  SphereTransition(double r_min, double r_max);

  double operator()(const std::array<double, 3>& source_coords) const override;
  DataVector operator()(
      const std::array<DataVector, 3>& source_coords) const override;

  std::optional<double> original_radius_over_radius(
      const std::array<double, 3>& target_coords,
      double distorted_radius) const override;

  double map_over_radius(
      const std::array<double, 3>& source_coords) const override;
  DataVector map_over_radius(
      const std::array<DataVector, 3>& source_coords) const override;

  std::array<double, 3> gradient(
      const std::array<double, 3>& source_coords) const override;
  std::array<DataVector, 3> gradient(
      const std::array<DataVector, 3>& source_coords) const override;

  WRAPPED_PUPable_decl_template(SphereTransition);
  explicit SphereTransition(CkMigrateMessage* const msg);
  void pup(PUP::er& p) override;

  std::unique_ptr<ShapeMapTransitionFunction> get_clone() const override {
    return std::make_unique<SphereTransition>(*this);
  }

  bool operator==(const ShapeMapTransitionFunction& other) const override;
  bool operator!=(const ShapeMapTransitionFunction& other) const override;

 private:
  template <typename T>
  T call_impl(const std::array<T, 3>& source_coords) const;

  template <typename T>
  T map_over_radius_impl(const std::array<T, 3>& source_coords) const;

  template <typename T>
  std::array<T, 3> gradient_impl(const std::array<T, 3>& source_coords) const;

  double r_min_{};
  double r_max_{};
  double a_{};
  double b_{};
  static constexpr double eps_{std::numeric_limits<double>::epsilon() * 100};
};
}  // namespace domain::CoordinateMaps::ShapeMapTransitionFunctions
