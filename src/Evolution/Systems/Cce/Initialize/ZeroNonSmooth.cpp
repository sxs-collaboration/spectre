// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/ZeroNonSmooth.hpp"

#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::InitializeJ {

ZeroNonSmooth::ZeroNonSmooth(const double angular_coordinate_tolerance,
                             const size_t max_iterations,
                             const bool require_convergence) noexcept
    : angular_coordinate_tolerance_{angular_coordinate_tolerance},
      max_iterations_{max_iterations},
      require_convergence_{require_convergence} {}

std::unique_ptr<InitializeJ<false>> ZeroNonSmooth::get_clone() const noexcept {
  return std::make_unique<ZeroNonSmooth>(angular_coordinate_tolerance_,
                                         max_iterations_);
}

void ZeroNonSmooth::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& /* boundary_dr_j*/,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& /*r*/, const size_t l_max,
    const size_t /*number_of_radial_points*/) const noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  const SpinWeighted<ComplexDataVector, 2> j_boundary_view;
  make_const_view(make_not_null(&j_boundary_view), get(*j), 0,
                  number_of_angular_points);

  get(*j).data() = 0.0;
  const double final_angular_coordinate_deviation =
      detail::adjust_angular_coordinates_for_j(
          j, cartesian_cauchy_coordinates, angular_cauchy_coordinates,
          get(boundary_j), l_max, angular_coordinate_tolerance_,
          max_iterations_, false);
  if (final_angular_coordinate_deviation > angular_coordinate_tolerance_ and
      require_convergence_) {
    ERROR(MakeString{}
          << "Initial data iterative angular solve did not reach "
             "target tolerance "
          << angular_coordinate_tolerance_ << ".\n"
          << "Exited after " << max_iterations_
          << " iterations, achieving final\n"
             "maximum over collocation points deviation of J from target of "
          << final_angular_coordinate_deviation);
  } else if (final_angular_coordinate_deviation >
             angular_coordinate_tolerance_) {
    Parallel::printf(
        "Warning: iterative angular solve did not reach "
        "target tolerance %e.\n"
        "Exited after %zu iterations, achieving final maximum over "
        "collocation points deviation of J from target of %e\n"
        "Proceeding with evolution using the partial result from partial "
        "angular solve.",
        angular_coordinate_tolerance_, max_iterations_,
        final_angular_coordinate_deviation);
  }
}

void ZeroNonSmooth::pup(PUP::er& p) noexcept {
  p | angular_coordinate_tolerance_;
  p | max_iterations_;
}

PUP::able::PUP_ID ZeroNonSmooth::my_PUP_ID = 0;
}  // namespace Cce::InitializeJ
