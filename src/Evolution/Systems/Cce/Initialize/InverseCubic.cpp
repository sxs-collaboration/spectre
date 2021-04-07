// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/InverseCubic.hpp"

#include <cstddef>
#include <memory>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::InitializeJ {

std::unique_ptr<InitializeJ> InverseCubic::get_clone() const
    noexcept {
  return std::make_unique<InverseCubic>();
}

void InverseCubic::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, const size_t l_max,
    const size_t number_of_radial_points) const noexcept {
  const DataVector one_minus_y_collocation =
      1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
                number_of_radial_points);
  for (size_t i = 0; i < number_of_radial_points; i++) {
    ComplexDataVector angular_view_j{
        get(*j).data().data() + get(boundary_j).size() * i,
        get(boundary_j).size()};
    // auto is acceptable here as these two values are only used once in the
    // below computation. `auto` causes an expression template to be
    // generated, rather than allocating.
    const auto one_minus_y_coefficient =
        0.25 * (3.0 * get(boundary_j).data() +
                get(r).data() * get(boundary_dr_j).data());
    const auto one_minus_y_cubed_coefficient =
        -0.0625 *
        (get(boundary_j).data() + get(r).data() * get(boundary_dr_j).data());
    angular_view_j =
        one_minus_y_collocation[i] * one_minus_y_coefficient +
        pow<3>(one_minus_y_collocation[i]) * one_minus_y_cubed_coefficient;
  }
  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto collocation_point : collocation) {
    get<0>(*angular_cauchy_coordinates)[collocation_point.offset] =
        collocation_point.theta;
    get<1>(*angular_cauchy_coordinates)[collocation_point.offset] =
        collocation_point.phi;
  }
  get<0>(*cartesian_cauchy_coordinates) =
      sin(get<0>(*angular_cauchy_coordinates)) *
      cos(get<1>(*angular_cauchy_coordinates));
  get<1>(*cartesian_cauchy_coordinates) =
      sin(get<0>(*angular_cauchy_coordinates)) *
      sin(get<1>(*angular_cauchy_coordinates));
  get<2>(*cartesian_cauchy_coordinates) =
      cos(get<0>(*angular_cauchy_coordinates));
}

void InverseCubic::pup(PUP::er& /*p*/) noexcept {}

PUP::able::PUP_ID InverseCubic::my_PUP_ID = 0;
}  // namespace Cce::InitializeJ
