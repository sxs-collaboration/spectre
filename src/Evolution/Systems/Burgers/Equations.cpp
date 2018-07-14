// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/Equations.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace Burgers {
void LocalLaxFriedrichsFlux::package_data(
    gsl::not_null<Variables<package_tags>*> packaged_data,
    const Scalar<DataVector>& normal_dot_flux_u,
    const Scalar<DataVector>& u) const noexcept {
  get<Tags::U>(*packaged_data) = u;
  get<::Tags::NormalDotFlux<Tags::U>>(*packaged_data) = normal_dot_flux_u;
}

void LocalLaxFriedrichsFlux::operator()(
    gsl::not_null<Scalar<DataVector>*> normal_dot_numerical_flux_u,
    const Scalar<DataVector>& normal_dot_flux_u_interior,
    const Scalar<DataVector>& u_interior,
    const Scalar<DataVector>& minus_normal_dot_flux_u_exterior,
    const Scalar<DataVector>& u_exterior) const noexcept {
  get(*normal_dot_numerical_flux_u) =
      0.5 *
      (get(normal_dot_flux_u_interior) - get(minus_normal_dot_flux_u_exterior));

  // Burgers is 1D, so the boundaries only have one point.
  const double local_max_speed =
      std::max(std::abs(get(u_interior)[0]), std::abs(get(u_exterior)[0]));
  get(*normal_dot_numerical_flux_u) +=
      0.5 * local_max_speed * (get(u_interior) - get(u_exterior));
}

double ComputeLargestCharacteristicSpeed::apply(
    const Scalar<DataVector>& u) noexcept {
  return max(abs(get(u)));
}
}  // namespace Burgers

template Variables<tmpl::list<
    Tags::div<Tags::Flux<Burgers::Tags::U, tmpl::size_t<1>, Frame::Inertial>>>>
divergence(
    const Variables<tmpl::list<
        Tags::Flux<Burgers::Tags::U, tmpl::size_t<1>, Frame::Inertial>>>& F,
    const Mesh<1>& mesh,
    const InverseJacobian<DataVector, 1, Frame::Logical,
                          Frame::Inertial>& inverse_jacobian) noexcept;
