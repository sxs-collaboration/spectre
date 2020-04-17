// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/Equations.hpp"

#include <algorithm>
#include <array>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/ScalarWave/System.hpp"
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
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_psi,
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::ij<DataVector, Dim, Frame::Inertial>& d_phi,
    const Scalar<DataVector>& gamma2) noexcept {
  get(*dt_psi) = -get(pi);
  get(*dt_pi) = -get<0, 0>(d_phi);
  for (size_t d = 1; d < Dim; ++d) {
    get(*dt_pi) -= d_phi.get(d, d);
  }
  for (size_t d = 0; d < Dim; ++d) {
    dt_phi->get(d) = -d_pi.get(d) + get(gamma2) * (d_psi.get(d) - phi.get(d));
  }
}

template <size_t Dim>
void ComputeNormalDotFluxes<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> pi_normal_dot_flux,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        phi_normal_dot_flux,
    const gsl::not_null<Scalar<DataVector>*> psi_normal_dot_flux,
    const Scalar<DataVector>& pi) noexcept {
  destructive_resize_components(pi_normal_dot_flux, get(pi).size());
  destructive_resize_components(phi_normal_dot_flux, get(pi).size());
  destructive_resize_components(psi_normal_dot_flux, get(pi).size());
  get(*pi_normal_dot_flux) = 0.0;
  get(*psi_normal_dot_flux) = 0.0;
  for (size_t i = 0; i < Dim; ++i) {
    phi_normal_dot_flux->get(i) = 0.0;
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
  template Variables<                                                        \
      db::wrap_tags_in<::Tags::deriv, derivative_tags<DIM(data)>,            \
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
