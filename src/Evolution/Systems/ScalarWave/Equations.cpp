// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/Equations.hpp"

#include <algorithm>
#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/ScalarWave/System.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"   // IWYU pragma: keep
#include "Utilities/TMPL.hpp"  // IWYU pragma: keep

// IWYU pragma: no_forward_declare Tensor

namespace ScalarWave {
// Doxygen is not good at templates and so we have to hide the definition.
/// \cond
template <size_t Dim>
void ComputeDuDt<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> dt_pi,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> dt_phi,
    const gsl::not_null<Scalar<DataVector>*> dt_psi,
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_pi,
    const tnsr::ij<DataVector, Dim, Frame::Inertial>& d_phi) noexcept {
  get(*dt_psi) = -get(pi);
  get(*dt_pi) = -get<0, 0>(d_phi);
  for (size_t d = 1; d < Dim; ++d) {
    get(*dt_pi) -= d_phi.get(d, d);
  }
  for (size_t d = 0; d < Dim; ++d) {
    dt_phi->get(d) = -d_pi.get(d);
  }
}

template <size_t Dim>
void ComputeNormalDotFluxes<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> pi_normal_dot_flux,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        phi_normal_dot_flux,
    const gsl::not_null<Scalar<DataVector>*> psi_normal_dot_flux,
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        interface_unit_normal) noexcept {
  // We assume that all values of psi_normal_dot_flux are the same. The reason
  // is that std::fill is actually surprisingly/disappointingly slow.
  if (psi_normal_dot_flux->get()[0] != 0.0) {
    std::fill(psi_normal_dot_flux->get().begin(),
              psi_normal_dot_flux->get().end(), 0.0);
  }

  get(*pi_normal_dot_flux) = get<0>(interface_unit_normal) * get<0>(phi);
  for (size_t i = 1; i < Dim; ++i) {
    get(*pi_normal_dot_flux) += interface_unit_normal.get(i) * phi.get(i);
  }

  for (size_t i = 0; i < Dim; ++i) {
    phi_normal_dot_flux->get(i) = interface_unit_normal.get(i) * get(pi);
  }
}

template <size_t Dim>
void UpwindFlux<Dim>::package_data(
    const gsl::not_null<Variables<package_tags>*> packaged_data,
    const Scalar<DataVector>& normal_dot_flux_pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_dot_flux_phi,
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
    const noexcept {
  // Computes the contribution to the numerical flux from one side of the
  // interface.
  //
  // The packaged_data stores:
  // <Pi> = pi
  // <NormalDotFlux<Pi>> = normal_dot_flux_pi
  // <NormalTimesFluxPi_i> = normal_dot_flux_pi * n_i
  // <NormalDotFlux<Phi>_i> = normal_dot_flux_phi_i
  //
  // Note: when Upwind::operator() is called, an Element passes in its own
  // packaged data to fill the interior fields, and its neighbors packaged data
  // to fill the exterior fields. This introduces a sign flip for each normal
  // used in computing the exterior fields.
  get<Pi>(*packaged_data) = pi;
  get<Tags::NormalDotFlux<Pi>>(*packaged_data) = normal_dot_flux_pi;
  get<Tags::NormalDotFlux<Phi<Dim>>>(*packaged_data) = normal_dot_flux_phi;
  auto& normal_times_flux_pi = get<NormalTimesFluxPi>(*packaged_data);
  for (size_t d = 0; d < Dim; ++d) {
    normal_times_flux_pi.get(d) =
        interface_unit_normal.get(d) * get(normal_dot_flux_pi);
  }
}

template <size_t Dim>
void UpwindFlux<Dim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> pi_normal_dot_numerical_flux,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        phi_normal_dot_numerical_flux,
    const gsl::not_null<Scalar<DataVector>*> psi_normal_dot_numerical_flux,
    const Scalar<DataVector>& normal_dot_flux_pi_interior,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        normal_dot_flux_phi_interior,
    const Scalar<DataVector>& pi_interior,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        normal_times_flux_pi_interior,
    const Scalar<DataVector>& minus_normal_dot_flux_pi_exterior,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        minus_normal_dot_flux_phi_exterior,
    const Scalar<DataVector>& pi_exterior,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        normal_times_flux_pi_exterior) const noexcept {
  std::fill(psi_normal_dot_numerical_flux->get().begin(),
            psi_normal_dot_numerical_flux->get().end(), 0.);
  get(*pi_normal_dot_numerical_flux) =
      0.5 *
      (get(pi_interior) - get(pi_exterior) + get(normal_dot_flux_pi_interior) -
       get(minus_normal_dot_flux_pi_exterior));
  for (size_t d = 0; d < Dim; ++d) {
    phi_normal_dot_numerical_flux->get(d) =
        0.5 * (normal_dot_flux_phi_interior.get(d) -
               minus_normal_dot_flux_phi_exterior.get(d) +
               normal_times_flux_pi_interior.get(d) -
               normal_times_flux_pi_exterior.get(d));
  }
}
/// \endcond
}  // namespace ScalarWave

// Generate explicit instantiations of partial_derivatives function as well as
// all other functions in Equations.cpp

#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"  // IWYU pragma: keep

template <size_t Dim>
using derivative_tags = typename ScalarWave::System<Dim>::gradients_tags;

template <size_t Dim>
using variables_tags =
    typename ScalarWave::System<Dim>::variables_tag::tags_list;

using derivative_frame = Frame::Inertial;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                               \
  template class ScalarWave::ComputeDuDt<DIM(data)>;                         \
  template class ScalarWave::ComputeNormalDotFluxes<DIM(data)>;              \
  template class ScalarWave::UpwindFlux<DIM(data)>;                          \
  template Variables<                                                        \
      db::wrap_tags_in<Tags::deriv, derivative_tags<DIM(data)>,              \
                       tmpl::size_t<DIM(data)>, derivative_frame>>           \
  partial_derivatives<derivative_tags<DIM(data)>, variables_tags<DIM(data)>, \
                      DIM(data), derivative_frame>(                          \
      const Variables<variables_tags<DIM(data)>>& u,                         \
      const Mesh<DIM(data)>& mesh,                                           \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,           \
                            derivative_frame>& inverse_jacobian) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
