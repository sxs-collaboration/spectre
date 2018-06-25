// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Equations.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Variables.hpp"      // IWYU pragma: keep
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"  // IWYU pragma: keep

template <typename X, typename Symm, typename IndexList>
class Tensor;

namespace CurvedScalarWave {
/// \cond
template <size_t Dim>
void ComputeDuDt<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> dt_pi,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> dt_phi,
    const gsl::not_null<Scalar<DataVector>*> dt_psi,
    const Scalar<DataVector>& pi, const tnsr::i<DataVector, Dim>& phi,
    const tnsr::i<DataVector, Dim>& d_psi, const tnsr::i<DataVector, Dim>& d_pi,
    const tnsr::ij<DataVector, Dim>& d_phi, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim>& shift,
    const tnsr::i<DataVector, Dim>& deriv_lapse,
    const tnsr::iJ<DataVector, Dim>& deriv_shift,
    const tnsr::II<DataVector, Dim>& upper_spatial_metric,
    const tnsr::I<DataVector, Dim>& trace_spatial_christoffel,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>& gamma1,
    const Scalar<DataVector>& gamma2) noexcept {
  dt_psi->get() = -lapse.get() * pi.get();
  for (size_t m = 0; m < Dim; ++m) {
    dt_psi->get() +=
        shift.get(m) *
        (d_psi.get(m) + gamma1.get() * (d_psi.get(m) - phi.get(m)));
  }

  dt_pi->get() = lapse.get() * pi.get() * trace_extrinsic_curvature.get();
  for (size_t m = 0; m < Dim; ++m) {
    dt_pi->get() += shift.get(m) * d_pi.get(m);
    dt_pi->get() += lapse.get() * phi.get(m) * trace_spatial_christoffel.get(m);
    dt_pi->get() += gamma1.get() * gamma2.get() * shift.get(m) *
                    (d_psi.get(m) - phi.get(m));
  }
  for (size_t m = 0; m < Dim; ++m) {
    for (size_t n = 0; n < Dim; ++n) {
      dt_pi->get() -=
          lapse.get() * upper_spatial_metric.get(m, n) * d_phi.get(m, n);
      dt_pi->get() -=
          upper_spatial_metric.get(m, n) * deriv_lapse.get(m) * phi.get(n);
    }
  }
  for (size_t k = 0; k < Dim; ++k) {
    dt_phi->get(k) =
        -lapse.get() *
            (d_pi.get(k) + gamma2.get() * (phi.get(k) - d_psi.get(k))) -
        pi.get() * deriv_lapse.get(k);
    for (size_t m = 0; m < Dim; ++m) {
      dt_phi->get(k) +=
          shift.get(m) * d_phi.get(m, k) + phi.get(m) * deriv_shift.get(k, m);
    }
  }
}
/// \endcond
}  // namespace CurvedScalarWave
// Generate explicit instantiations of partial_derivatives function as well as
// all other functions in Equations.cpp

#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"  // IWYU pragma: keep

template <size_t Dim>
using derivative_tags = typename CurvedScalarWave::System<Dim>::gradients_tags;

template <size_t Dim>
using variables_tags =
    typename CurvedScalarWave::System<Dim>::variables_tag::tags_list;

using derivative_frame = Frame::Inertial;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                               \
  template class CurvedScalarWave::ComputeDuDt<DIM(data)>;                   \
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
