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
 * \brief Initialize \f$J\f$ on the first hypersurface to be vanishing, finding
 * the appropriate angular coordinates to be continuous with the provided
 * worldtube boundary data.
 *
 * \details Internally, this performs an iterative solve for the angular
 * coordinates necessary to give rise to a vanishing gauge-transformed J on the
 * worldtube boundary. The parameters for the iterative procedure are determined
 * by options `ZeroNonSmooth::AngularCoordinateTolerance` and
 * `ZeroNonSmooth::MaxIterations`. The resulting `J` will necessarily
 * have vanishing first radial derivative, and so will typically not be smooth
 * (only continuous) with the provided Cauchy data at the worldtube boundary.
 */
struct ZeroNonSmooth : InitializeJ<false> {
  struct AngularCoordinateTolerance {
    using type = double;
    static std::string name() { return "AngularCoordTolerance"; }
    static constexpr Options::String help = {
        "Tolerance of initial angular coordinates for CCE"};
    static type lower_bound() { return 1.0e-14; }
    static type upper_bound() { return 1.0e-3; }
    static type suggested_value() { return 1.0e-10; }
  };

  struct MaxIterations {
    using type = size_t;
    static constexpr Options::String help = {
        "Number of linearized inversion iterations."};
    static type lower_bound() { return 10; }
    static type upper_bound() { return 1000; }
    static type suggested_value() { return 300; }
  };

  struct RequireConvergence {
    using type = bool;
    static constexpr Options::String help = {
        "If true, initialization will error if it hits MaxIterations"};
    static type suggested_value() { return true; }
  };
  using options =
      tmpl::list<AngularCoordinateTolerance, MaxIterations, RequireConvergence>;

  static constexpr Options::String help = {
      "Initialization process where J is set so Psi0 is vanishing\n"
      "(roughly a no incoming radiation condition)"};

  WRAPPED_PUPable_decl_template(ZeroNonSmooth);  // NOLINT
  explicit ZeroNonSmooth(CkMigrateMessage* /*unused*/) {}

  ZeroNonSmooth(double angular_coordinate_tolerance, size_t max_iterations,
                bool require_convergence = false);

  ZeroNonSmooth() = default;

  std::unique_ptr<InitializeJ> get_clone() const override;

  void operator()(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
      gsl::not_null<
          tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
          angular_cauchy_coordinates,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, size_t l_max,
      size_t number_of_radial_points) const override;

  void pup(PUP::er& p) override;

 private:
  double angular_coordinate_tolerance_ = 1.0e-10;
  size_t max_iterations_ = 300;
  bool require_convergence_ = false;
};
}  // namespace InitializeJ
}  // namespace Cce
