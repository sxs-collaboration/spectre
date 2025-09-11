// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace GrSelfForce::BoundaryConditions {

/*!
 * \brief Radial Sommerfeld boundary conditions for the m-mode metric
 * perturbations.
 *
 * These radial boundary conditions enforce an outgoing direction of radiation
 * propagation. They apply both near the Kerr horizon (inner radial
 * boundary) and at large distance (outer radial boundary):
 *
 * \begin{equation}
 * n_i F^i = i m \Omega \Psi_m
 * \end{equation}
 *
 * These boundary conditions currently assume a circular equatorial orbit.
 */
class Sommerfeld : public elliptic::BoundaryConditions::BoundaryCondition<2> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<2>;
  using GradTensorType =
      TensorMetafunctions::prepend_spatial_index<tnsr::aa<ComplexDataVector, 3>,
                                                 2, UpLo::Lo, Frame::Inertial>;

 public:
  struct BlackHoleMass {
    static constexpr Options::String help =
        "Kerr mass parameter 'M' of the black hole";
    using type = double;
  };
  struct BlackHoleSpin {
    static constexpr Options::String help =
        "Kerr dimensionless spin parameter 'chi' of the black hole";
    using type = double;
  };
  struct OrbitalRadius {
    static constexpr Options::String help =
        "Radius 'r_0' of the circular orbit";
    using type = double;
  };
  struct MModeNumber {
    static constexpr Options::String help =
        "Mode number 'm' of the scalar field";
    using type = int;
  };

  static constexpr Options::String help =
      "Radial Sommerfeld boundary condition";
  using options =
      tmpl::list<BlackHoleMass, BlackHoleSpin, OrbitalRadius, MModeNumber>;

  Sommerfeld() = default;
  Sommerfeld(const Sommerfeld&) = default;
  Sommerfeld& operator=(const Sommerfeld&) = default;
  Sommerfeld(Sommerfeld&&) = default;
  Sommerfeld& operator=(Sommerfeld&&) = default;
  ~Sommerfeld() override = default;

  explicit Sommerfeld(double black_hole_mass, double black_hole_spin,
                      double orbital_radius, int m_mode_number);

  double black_hole_mass() const { return black_hole_mass_; }
  double black_hole_spin() const { return black_hole_spin_; }
  double orbital_radius() const { return orbital_radius_; }
  int m_mode_number() const { return m_mode_number_; }

  /// \cond
  explicit Sommerfeld(CkMigrateMessage* m);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Sommerfeld);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override;

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    return {elliptic::BoundaryConditionType::Neumann};
  }

  using argument_tags = tmpl::list<>;
  using volume_tags = tmpl::list<>;

  void apply(
      gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> field,
      gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> n_dot_field_gradient,
      const GradTensorType& deriv_field) const;

  using argument_tags_linearized = tmpl::list<>;
  using volume_tags_linearized = tmpl::list<>;

  void apply_linearized(
      gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> field_correction,
      gsl::not_null<tnsr::aa<ComplexDataVector, 3>*>
          n_dot_field_correction_gradient,
      const GradTensorType& deriv_field_correction) const;

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override;

 private:
  friend bool operator==(const Sommerfeld& lhs, const Sommerfeld& rhs);

  double black_hole_mass_{std::numeric_limits<double>::signaling_NaN()};
  double black_hole_spin_{std::numeric_limits<double>::signaling_NaN()};
  double orbital_radius_{std::numeric_limits<double>::signaling_NaN()};
  int m_mode_number_{};
};

bool operator!=(const Sommerfeld& lhs, const Sommerfeld& rhs);

}  // namespace GrSelfForce::BoundaryConditions
