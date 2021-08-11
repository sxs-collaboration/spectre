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
 * \brief A transition function that falls off as \f$f(r) = (ar + b) / r\f$. The
 * coefficients \f$a\f$ and \f$b\f$ are chosen so that the map falls off
 * linearly from 1 at `r_min` to 0 at `r_max`.
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

  bool operator==(const ShapeMapTransitionFunction& other) const override;
  bool operator!=(const ShapeMapTransitionFunction& other) const override;

 private:
  template <typename T>
  T call_impl(const std::array<T, 3>& source_coords) const;

  template <typename T>
  T map_over_radius_impl(const std::array<T, 3>& source_coords) const;

  template <typename T>
  std::array<T, 3> gradient_impl(const std::array<T, 3>& source_coords) const;

  // checks that the magnitudes are all between `r_min_` and `r_max_`
  template <typename T>
  void check_magnitudes(const T& mag) const;

  double r_min_{};
  double r_max_{};
  double a_{};
  double b_{};
};
}  // namespace domain::CoordinateMaps::ShapeMapTransitionFunctions
