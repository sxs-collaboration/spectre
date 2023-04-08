// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/ZeroNonSmooth.hpp"

#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::InitializeJ {

ZeroNonSmooth::ZeroNonSmooth(const double angular_coordinate_tolerance,
                             const size_t max_iterations,
                             const bool require_convergence)
    : angular_coordinate_tolerance_{angular_coordinate_tolerance},
      max_iterations_{max_iterations},
      require_convergence_{require_convergence} {}

std::unique_ptr<InitializeJ<false>> ZeroNonSmooth::get_clone() const {
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
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& /*r*/,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& /*beta*/,
    const size_t l_max, const size_t /*number_of_radial_points*/,
    const gsl::not_null<Parallel::NodeLock*> /*hdf5_lock*/) const {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  const SpinWeighted<ComplexDataVector, 2> j_boundary_view;
  make_const_view(make_not_null(&j_boundary_view), get(*j), 0,
                  number_of_angular_points);

  get(*j).data() = 0.0;

  Variables<
      tmpl::list<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 2>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                                      std::integral_constant<int, 0>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexDataVector>,
                                      std::integral_constant<int, 0>>>>
      iteration_buffers{number_of_angular_points};

  auto& evolution_gauge_surface_j =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                   std::integral_constant<int, 2>>>(
          iteration_buffers));
  auto& interpolated_k =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          iteration_buffers));
  auto& gauge_omega =
      get<::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          iteration_buffers);

  // find a coordinate transformation such that in the new coordinates,
  // J is zero at the worldtube (as we have set it to above)
  auto iteration_function =
      [&interpolated_k, &gauge_omega, &evolution_gauge_surface_j, &boundary_j](
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
              gauge_c_step,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
              gauge_d_step,
          const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
          const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
          const Spectral::Swsh::SwshInterpolator& iteration_interpolator) {
        iteration_interpolator.interpolate(
            make_not_null(&evolution_gauge_surface_j), get(boundary_j));
        interpolated_k.data() =
            sqrt(1.0 + evolution_gauge_surface_j.data() *
                           conj(evolution_gauge_surface_j.data()));
        get(gauge_omega).data() =
            0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                       get(gauge_c).data() * conj(get(gauge_c).data()));
        evolution_gauge_surface_j.data() =
            0.25 *
            (square(conj(get(gauge_d).data())) *
                 evolution_gauge_surface_j.data() +
             square(get(gauge_c).data()) *
                 conj(evolution_gauge_surface_j.data()) +
             2.0 * get(gauge_c).data() * conj(get(gauge_d).data()) *
                 interpolated_k.data()) /
            square(get(gauge_omega).data());

        double max_error = max(abs(evolution_gauge_surface_j.data()));

        // The alteration in each of the spin-weighted Jacobian factors
        // determined by linearizing the system in small J
        get(*gauge_c_step).data() =
            -0.5 * evolution_gauge_surface_j.data() *
            square(get(gauge_omega).data()) /
            (get(gauge_d).data() * interpolated_k.data());
        get(*gauge_d_step).data() = get(*gauge_c_step).data() *
                                    conj(get(gauge_c).data()) /
                                    conj(get(gauge_d).data());
        return max_error;
      };

  detail::iteratively_adapt_angular_coordinates(
      cartesian_cauchy_coordinates, angular_cauchy_coordinates, l_max,
      angular_coordinate_tolerance_, max_iterations_, 1.0e-2,
      iteration_function, require_convergence_);
}

void ZeroNonSmooth::pup(PUP::er& p) {
  p | angular_coordinate_tolerance_;
  p | max_iterations_;
}

PUP::able::PUP_ID ZeroNonSmooth::my_PUP_ID = 0;
}  // namespace Cce::InitializeJ
