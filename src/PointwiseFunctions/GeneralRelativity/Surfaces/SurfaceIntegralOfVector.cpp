// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Surfaces/SurfaceIntegralOfVector.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace StrahlkorperGr {
template <typename Frame>
double euclidean_surface_integral_of_vector(
    const Scalar<DataVector>& area_element,
    const tnsr::I<DataVector, 3, Frame>& vector,
    const tnsr::i<DataVector, 3, Frame>& normal_one_form,
    const Strahlkorper<Frame>& strahlkorper) {
  const DataVector integrand =
      get(area_element) * get(dot_product(vector, normal_one_form)) /
      sqrt(get(dot_product(normal_one_form, normal_one_form)));
  return strahlkorper.ylm_spherepack().definite_integral(integrand.data());
}
}  // namespace StrahlkorperGr

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                            \
  template double StrahlkorperGr::euclidean_surface_integral_of_vector( \
      const Scalar<DataVector>& area_element,                           \
      const tnsr::I<DataVector, 3, FRAME(data)>& vector,                \
      const tnsr::i<DataVector, 3, FRAME(data)>& normal_one_form,       \
      const Strahlkorper<FRAME(data)>& strahlkorper);
GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Grid, Frame::Distorted, Frame::Inertial))
#undef INSTANTIATE
#undef FRAME
