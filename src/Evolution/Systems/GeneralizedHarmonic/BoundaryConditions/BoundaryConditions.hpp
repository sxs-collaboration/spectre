// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace GeneralizedHarmonic {
namespace BoundaryConditions {}  // namespace BoundaryConditions

namespace BoundaryConditions {
namespace Bjorhus {}  // namespace Bjorhus

namespace Bjorhus {
enum class VSpacetimeMetricBcMethod { Freezing, ConstraintPreserving, Unknown };
enum class VZeroBcMethod { Freezing, ConstraintPreserving, Unknown };
enum class VPlusBcMethod { Freezing, Unknown };
enum class VMinusBcMethod {
  Freezing,
  ConstraintPreserving,
  ConstraintPreservingPhysical,
  Unknown
};

struct Freezing {
  static const VSpacetimeMetricBcMethod v_spacetime_bc_method =
      VSpacetimeMetricBcMethod::Freezing;
  static const VZeroBcMethod v_zero_bc_method = VZeroBcMethod::Freezing;
  static const VPlusBcMethod v_plus_bc_method = VPlusBcMethod::Freezing;
  static const VMinusBcMethod v_minus_bc_method = VMinusBcMethod::Freezing;
};

struct ConstraintPreserving {
  static const VSpacetimeMetricBcMethod v_spacetime_bc_method =
      VSpacetimeMetricBcMethod::ConstraintPreserving;
  static const VZeroBcMethod v_zero_bc_method =
      VZeroBcMethod::ConstraintPreserving;
  // (This field is never incoming, which is why we use Freezing)
  static const VPlusBcMethod v_plus_bc_method = VPlusBcMethod::Freezing;
  static const VMinusBcMethod v_minus_bc_method =
      VMinusBcMethod::ConstraintPreserving;
};

struct ConstraintPreservingPhysical {
  static const VSpacetimeMetricBcMethod v_spacetime_bc_method =
      VSpacetimeMetricBcMethod::ConstraintPreserving;
  static const VZeroBcMethod v_zero_bc_method =
      VZeroBcMethod::ConstraintPreserving;
  // (This field is never incoming, which is why we use Freezing)
  static const VPlusBcMethod v_plus_bc_method = VPlusBcMethod::Freezing;
  static const VMinusBcMethod v_minus_bc_method =
      VMinusBcMethod::ConstraintPreservingPhysical;
};
}  // namespace Bjorhus
}  // namespace BoundaryConditions
}  // namespace GeneralizedHarmonic
