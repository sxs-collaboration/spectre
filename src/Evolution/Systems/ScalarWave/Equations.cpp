// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/Equations.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/ScalarWave/System.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarWave {
template <size_t Dim>
void ComputeDuDt<Dim>::apply(
    gsl::not_null<Scalar<DataVector>*> dt_pi,
    gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> dt_phi,
    gsl::not_null<Scalar<DataVector>*> dt_psi, const Scalar<DataVector>& pi,
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
}  // namespace ScalarWave

// Generate explicit instantiations of partial_derivatives function as well as
// all other functions in Equations.cpp

#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.cpp"

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
      db::wrap_tags_in<Tags::deriv, derivative_tags<DIM(data)>,              \
                       tmpl::size_t<DIM(data)>, derivative_frame>>           \
  partial_derivatives<derivative_tags<DIM(data)>, variables_tags<DIM(data)>, \
                      DIM(data), derivative_frame>(                          \
      const Variables<variables_tags<DIM(data)>>& u,                         \
      const Index<DIM(data)>& extents,                                       \
      const Tensor<                                                          \
          DataVector, tmpl::integral_list<std::int32_t, 2, 1>,               \
          tmpl::list<SpatialIndex<DIM(data), UpLo::Up, Frame::Logical>,      \
                     SpatialIndex<DIM(data), UpLo::Lo, derivative_frame>>>&  \
          inverse_jacobian) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
