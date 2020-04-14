// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/NoIncomingRadiation.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "NumericalAlgorithms/OdeIntegration/OdeIntegration.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/Printf.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace InitializeJ {
namespace detail {

void radial_evolve_psi0_condition(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> volume_j_id,
    const SpinWeighted<ComplexDataVector, 2>& boundary_j,
    const SpinWeighted<ComplexDataVector, 2>& boundary_dr_j,
    const SpinWeighted<ComplexDataVector, 0>& r, const size_t l_max,
    const size_t number_of_radial_points) noexcept {
  // use the maximum to measure the scale for the vector quantities
  const double j_scale = max(abs(boundary_j.data()));
  const double dy_j_scale = max(abs(0.5 * boundary_dr_j.data() * r.data()));
  // set initial step size according to the first couple of steps in section
  // II.4 of Solving Ordinary Differential equations by Hairer, Norsett, and
  // Wanner
  double initial_radial_step = 1.0e-6;
  if (j_scale > 1.0e-5 and dy_j_scale > 1.0e-5) {
    initial_radial_step = 0.01 * j_scale / dy_j_scale;
  }

  const auto psi_0_condition_system =
      [](const std::array<ComplexDataVector, 2>& bondi_j_and_i,
         std::array<ComplexDataVector, 2>& dy_j_and_dy_i,
         const double y) noexcept {
        dy_j_and_dy_i[0] = bondi_j_and_i[1];
        const auto& bondi_j = bondi_j_and_i[0];
        const auto& bondi_i = bondi_j_and_i[1];
        dy_j_and_dy_i[1] =
            -0.0625 *
            (square(conj(bondi_i) * bondi_j) + square(conj(bondi_j) * bondi_i) -
             2.0 * bondi_i * conj(bondi_i) * (2.0 + bondi_j * conj(bondi_j))) *
            (4.0 * bondi_j + bondi_i * (1.0 - y)) /
            (1.0 + bondi_j * conj(bondi_j));
      };

  boost::numeric::odeint::dense_output_runge_kutta<
      boost::numeric::odeint::controlled_runge_kutta<
          boost::numeric::odeint::runge_kutta_dopri5<
              std::array<ComplexDataVector, 2>>>>
      dense_stepper = boost::numeric::odeint::make_dense_output(
          1.0e-14, 1.0e-14,
          boost::numeric::odeint::runge_kutta_dopri5<
              std::array<ComplexDataVector, 2>>{});
  dense_stepper.initialize(
      std::array<ComplexDataVector, 2>{
          {boundary_j.data(), 0.5 * boundary_dr_j.data() * r.data()}},
      -1.0, initial_radial_step);
  auto state_buffer =
      std::array<ComplexDataVector, 2>{{ComplexDataVector{boundary_j.size()},
                                        ComplexDataVector{boundary_j.size()}}};

  std::pair<double, double> step_range =
      dense_stepper.do_step(psi_0_condition_system);
  const auto& y_collocation =
      Spectral::collocation_points<Spectral::Basis::Legendre,
                                   Spectral::Quadrature::GaussLobatto>(
                                       number_of_radial_points);
  for (size_t y_collocation_point = 0;
       y_collocation_point < number_of_radial_points; ++y_collocation_point) {
    while(step_range.second < y_collocation[y_collocation_point]) {
      step_range = dense_stepper.do_step(psi_0_condition_system);
    }
    if (step_range.second < y_collocation[y_collocation_point] or
        step_range.first > y_collocation[y_collocation_point]) {
      ERROR(
          "Psi 0 radial integration failed. The current y value is "
          "incompatible with the required Gauss-Lobatto point.");
    }
    dense_stepper.calc_state(y_collocation[y_collocation_point], state_buffer);
    ComplexDataVector angular_view{
        volume_j_id->data().data() +
            y_collocation_point *
                Spectral::Swsh::number_of_swsh_collocation_points(l_max),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    angular_view = state_buffer[0];
  }
}
}  // namespace detail

NoIncomingRadiation::NoIncomingRadiation(
    const double angular_coordinate_tolerance, const size_t max_iterations,
    const bool require_convergence) noexcept
    : require_convergence_{require_convergence},
      angular_coordinate_tolerance_{angular_coordinate_tolerance},
      max_iterations_{max_iterations} {}

std::unique_ptr<InitializeJ> NoIncomingRadiation::get_clone() const noexcept {
  return std::make_unique<NoIncomingRadiation>(*this);
}

void NoIncomingRadiation::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, const size_t l_max,
    const size_t number_of_radial_points) const noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  detail::radial_evolve_psi0_condition(make_not_null(&get(*j)), get(boundary_j),
                                       get(boundary_dr_j), get(r), l_max,
                                       number_of_radial_points);
  const SpinWeighted<ComplexDataVector, 2> j_at_scri_view;
  make_const_view(make_not_null(&j_at_scri_view), get(*j),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const double final_angular_coordinate_deviation =
      detail::adjust_angular_coordinates_for_j(
          j, cartesian_cauchy_coordinates, angular_cauchy_coordinates,
          j_at_scri_view, l_max, angular_coordinate_tolerance_, max_iterations_,
          true);

  if (final_angular_coordinate_deviation > angular_coordinate_tolerance_ and
      require_convergence_) {
    ERROR(
        "Initial data iterative angular solve did not reach "
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

void NoIncomingRadiation::pup(PUP::er& p) noexcept {
  p | require_convergence_;
  p | angular_coordinate_tolerance_;
  p | max_iterations_;
}

/// \cond
PUP::able::PUP_ID NoIncomingRadiation::my_PUP_ID = 0;
/// \endcond
}  // namespace InitializeJ
}  // namespace Cce
