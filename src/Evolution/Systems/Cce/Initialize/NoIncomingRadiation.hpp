// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class ComplexDataVector;
/// \endcond

namespace Cce {
namespace InitializeJ {

/*!
 * \brief Initialize \f$J\f$ on the first hypersurface by constraining
 * \f$\Psi_0 = 0\f$.
 *
 * \details This algorithm first radially evolves the \f$\Psi_0 = 0\f$
 * condition, which can be converted to a second-order radial ODE for J. Then,
 * the initial data generator performs an iterative solve for the angular
 * coordinates necessary to ensure asymptotic flatness. The parameters for the
 * iterative procedure are determined by options
 * `AngularCoordinateTolerance` and `MaxIterations`.
 */
struct NoIncomingRadiation : InitializeJ<false> {
  struct AngularCoordinateTolerance {
    using type = double;
    static std::string name() noexcept { return "AngularCoordTolerance"; }
    static constexpr Options::String help = {
        "Tolerance of initial angular coordinates for CCE"};
    static type lower_bound() noexcept { return 1.0e-14; }
    static type upper_bound() noexcept { return 1.0e-3; }
    static type suggested_value() noexcept { return 1.0e-10; }
  };

  struct MaxIterations {
    using type = size_t;
    static constexpr Options::String help = {
        "Number of linearized inversion iterations."};
    static type lower_bound() noexcept { return 10; }
    static type upper_bound() noexcept { return 1000; }
    static type suggested_value() noexcept { return 300; }
  };

  struct RequireConvergence {
    using type = bool;
    static constexpr Options::String help = {
      "If true, initialization will error if it hits MaxIterations"};
    static type suggested_value() noexcept { return true; }
  };

  using options =
      tmpl::list<AngularCoordinateTolerance, MaxIterations, RequireConvergence>;
  static constexpr Options::String help = {
      "Initialization process where J is set so Psi0 is vanishing\n"
      "(roughly a no incoming radiation condition)"};

  WRAPPED_PUPable_decl_template(NoIncomingRadiation);  // NOLINT
  explicit NoIncomingRadiation(CkMigrateMessage* /*unused*/) noexcept {}

  NoIncomingRadiation(double angular_coordinate_tolerance,
                      size_t max_iterations,
                      bool require_convergence = false) noexcept;

  NoIncomingRadiation() = default;

  std::unique_ptr<InitializeJ> get_clone() const noexcept override;

  void operator()(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
      gsl::not_null<
          tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
          angular_cauchy_coordinates,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, size_t l_max,
      size_t number_of_radial_points) const noexcept override;

  void pup(PUP::er& p) noexcept override;

 private:
  bool require_convergence_ = false;
  double angular_coordinate_tolerance_ = 1.0e-10;
  size_t max_iterations_ = 300;
};
}  // namespace InitializeJ
}  // namespace Cce
