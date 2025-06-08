// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Surfaces/SurfaceIntegralOfScalar.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace gr::surfaces {
template <typename Frame>
double surface_integral_of_scalar(
    const Scalar<DataVector>& area_element, const Scalar<DataVector>& scalar,
    const ylm::Strahlkorper<Frame>& strahlkorper) {
  const DataVector integrand = get(area_element) * get(scalar);
  return strahlkorper.ylm_spherepack().definite_integral(integrand.data());
}

template <typename Frame>
double surface_average_of_scalar(const Scalar<DataVector>& area_element,
                                 const Scalar<DataVector>& scalar,
                                 const ylm::Strahlkorper<Frame>& strahlkorper) {
  const DataVector integrand = get(area_element) * get(scalar);

  const double surface_integral =
      strahlkorper.ylm_spherepack().definite_integral(integrand.data());
  const double surface_area =
      strahlkorper.ylm_spherepack().definite_integral(get(area_element).data());
  return surface_integral / surface_area;
}
}  // namespace gr::surfaces

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                \
  template double gr::surfaces::surface_integral_of_scalar( \
      const Scalar<DataVector>& area_element,               \
      const Scalar<DataVector>& scalar,                     \
      const ylm::Strahlkorper<FRAME(data)>& strahlkorper);  \
  template double gr::surfaces::surface_average_of_scalar(  \
      const Scalar<DataVector>& area_element,               \
      const Scalar<DataVector>& scalar,                     \
      const ylm::Strahlkorper<FRAME(data)>& strahlkorper);
GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Grid, Frame::Distorted, Frame::Inertial))
#undef INSTANTIATE
#undef FRAME
