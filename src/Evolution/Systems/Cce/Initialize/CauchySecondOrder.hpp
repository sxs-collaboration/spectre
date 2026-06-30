// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <string>

#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class ComplexDataVector;
/// \endcond

namespace Cce::InitializeJ {

/*!
 * \brief Initialize \f$J\f$ on the first hypersurface using a second-order
 * matching at the worldtube.
 *
 * \details The volume \f$J\f$ is built from the worldtube values of
 * \f$J\f$, \f$\partial_r J\f$, and \f$\partial_y^2 J\f$ computed from the
 * H hypersurface equation. The remaining angular coordinates are determined
 * iteratively to ensure asymptotic flatness. As a safeguard, the
 * initialization aborts if the second radial derivative of \f$J\f$ at scri+
 * of the final solution exceeds `MaxScriSecondDerivative`.
 */
struct CauchySecondOrder : InitializeJ<false> {
  struct AngularCoordinateTolerance {
    using type = double;
    static std::string name() { return "AngularCoordTolerance"; }
    static constexpr Options::String help = {
        "Tolerance of initial angular coordinates for CCE"};
    static type lower_bound() { return 1.0e-14; }
    static type upper_bound() { return 1.0e-3; }
    static type suggested_value() { return 1.0e-12; }
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

  struct MaxScriSecondDerivative {
    using type = double;
    static constexpr Options::String help = {
        "Abort initialization if the largest second radial derivative of J at "
        "scri+ of the final initial data exceeds this threshold. The "
        "second-order construction drives this derivative to (near) zero, so a "
        "large value indicates a poorly matched solution. Set to a large value "
        "to effectively disable the check."};
    static type lower_bound() { return 1.0e-14; }
    static type upper_bound() { return 1.0e2; }
    static type suggested_value() { return 1.0e-8; }
  };

  using options = tmpl::list<AngularCoordinateTolerance, MaxIterations,
                             RequireConvergence, MaxScriSecondDerivative>;
  static constexpr Options::String help = {
      "Second-order initial data generator for the Cauchy CCE evolution."};

  WRAPPED_PUPable_decl_template(CauchySecondOrder);  // NOLINT
  explicit CauchySecondOrder(CkMigrateMessage* /*unused*/) {}

  CauchySecondOrder(double angular_coordinate_tolerance, size_t max_iterations,
                    bool require_convergence,
                    double max_scri_second_derivative);

  CauchySecondOrder() = default;

  std::unique_ptr<InitializeJ> get_clone() const override;

  // Per-class tag lists. The flexible dispatch in `InitializeJ<false>` reads
  // these via `call_with_dynamic_type` so this generator can request more
  // worldtube boundary values than the simpler sibling classes do.
  using return_tags = tmpl::list<Tags::BondiJ, Tags::CauchyCartesianCoords,
                                 Tags::CauchyAngularCoords>;
  using argument_tags = tmpl::list<
      Tags::BoundaryValue<Tags::BondiJ>, Tags::BoundaryValue<Tags::BondiU>,
      Tags::BoundaryValue<Tags::BondiW>, Tags::BoundaryValue<Tags::BondiBeta>,
      Tags::BoundaryValue<Tags::BondiQ>,
      Tags::BoundaryValue<Tags::Du<Tags::BondiJ>>,
      Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>,
      Tags::BoundaryValue<Tags::Du<Tags::Dr<Tags::BondiJ>>>,
      Tags::BoundaryValue<Tags::Du<Tags::BondiR>>,
      Tags::BoundaryValue<Tags::BondiR>, Tags::LMax,
      Tags::NumberOfRadialPoints>;

  void operator()(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
      gsl::not_null<
          tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
          angular_cauchy_coordinates,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& boundary_u,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_w,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& boundary_q,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_du_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_du_dr_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_du_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, size_t l_max,
      size_t number_of_radial_points,
      gsl::not_null<Parallel::NodeLock*> hdf5_lock) const;

  void pup(PUP::er& p) override;

 private:
  bool require_convergence_ = true;
  double angular_coordinate_tolerance_ =
      std::numeric_limits<double>::signaling_NaN();
  size_t max_iterations_ = 0;
  double max_scri_second_derivative_ =
      std::numeric_limits<double>::signaling_NaN();
};
}  // namespace Cce::InitializeJ
