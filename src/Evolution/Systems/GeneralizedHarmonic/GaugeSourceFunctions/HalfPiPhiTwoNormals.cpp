// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/HalfPiPhiTwoNormals.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace gh::gauges {
template <size_t Dim, typename Frame>
void half_pi_and_phi_two_normals(
    const gsl::not_null<Scalar<DataVector>*> half_pi_two_normals,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame>*> half_phi_two_normals,
    const tnsr::A<DataVector, Dim, Frame>& spacetime_normal_vector,
    const tnsr::aa<DataVector, Dim, Frame>& pi,
    const tnsr::iaa<DataVector, Dim, Frame>& phi) {
  destructive_resize_components(half_pi_two_normals, get<0, 0>(pi).size());
  destructive_resize_components(half_phi_two_normals, get<0, 0>(pi).size());
  get(*half_pi_two_normals) = 0.0;
  for (size_t i = 0; i < Dim; ++i) {
    half_phi_two_normals->get(i) = 0.0;
  }
  for (size_t a = 0; a < Dim + 1; ++a) {
    get(*half_pi_two_normals) += spacetime_normal_vector.get(a) *
                                 spacetime_normal_vector.get(a) * pi.get(a, a);
    for (size_t i = 0; i < Dim; ++i) {
      half_phi_two_normals->get(i) += 0.5 * spacetime_normal_vector.get(a) *
                                      spacetime_normal_vector.get(a) *
                                      phi.get(i, a, a);
    }
    for (size_t b = a + 1; b < Dim + 1; ++b) {
      get(*half_pi_two_normals) += 2.0 * spacetime_normal_vector.get(a) *
                                   spacetime_normal_vector.get(b) *
                                   pi.get(a, b);
      for (size_t i = 0; i < Dim; ++i) {
        half_phi_two_normals->get(i) += spacetime_normal_vector.get(a) *
                                        spacetime_normal_vector.get(b) *
                                        phi.get(i, a, b);
      }
    }
  }
  get(*half_pi_two_normals) *= 0.5;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                            \
  template void half_pi_and_phi_two_normals(                            \
      const gsl::not_null<Scalar<DataVector>*> half_pi_two_normals,     \
      const gsl::not_null<tnsr::i<DataVector, DIM(data), FRAME(data)>*> \
          half_phi_two_normals,                                         \
      const tnsr::A<DataVector, DIM(data), FRAME(data)>&                \
          spacetime_normal_vector,                                      \
      const tnsr::aa<DataVector, DIM(data), FRAME(data)>& pi,           \
      const tnsr::iaa<DataVector, DIM(data), FRAME(data)>& phi);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Inertial, Frame::Grid))

#undef INSTANTIATE
#undef FRAME
#undef DIM
}  // namespace gh::gauges
