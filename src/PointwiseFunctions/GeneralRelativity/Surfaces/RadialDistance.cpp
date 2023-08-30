// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Surfaces/RadialDistance.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace gr::surfaces {
template <typename Frame>
void radial_distance(const gsl::not_null<Scalar<DataVector>*> radial_distance,
                     const ylm::Strahlkorper<Frame>& strahlkorper_a,
                     const ylm::Strahlkorper<Frame>& strahlkorper_b) {
  if (strahlkorper_a.expansion_center() != strahlkorper_b.expansion_center()) {
    ERROR(
        "Currently computing the radial distance between two Strahlkorpers "
        "is only supported if they have the same centers, but the "
        "strahlkorpers provided have centers "
        << strahlkorper_a.expansion_center() << " and "
        << strahlkorper_b.expansion_center());
  }
  get(*radial_distance)
      .destructive_resize(strahlkorper_a.ylm_spherepack().physical_size());
  if (strahlkorper_a.l_max() == strahlkorper_b.l_max() and
      strahlkorper_a.m_max() == strahlkorper_b.m_max()) {
    get(*radial_distance) =
        get(ylm::radius(strahlkorper_a)) - get(ylm::radius(strahlkorper_b));
  } else if (strahlkorper_a.l_max() > strahlkorper_b.l_max() or
             (strahlkorper_a.l_max() == strahlkorper_b.l_max() and
              strahlkorper_a.m_max() > strahlkorper_b.m_max())) {
    get(*radial_distance) =
        get(ylm::radius(strahlkorper_a)) -
        get(ylm::radius(ylm::Strahlkorper<Frame>(
            strahlkorper_a.l_max(), strahlkorper_a.m_max(), strahlkorper_b)));
  } else {
    get(*radial_distance) =
        -get(ylm::radius(strahlkorper_b)) +
        get(ylm::radius(ylm::Strahlkorper<Frame>(
            strahlkorper_b.l_max(), strahlkorper_b.m_max(), strahlkorper_a)));
  };
}
}  // namespace gr::surfaces

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                    \
  template void gr::surfaces::radial_distance<FRAME(data)>(     \
      const gsl::not_null<Scalar<DataVector>*> radial_distance, \
      const ylm::Strahlkorper<FRAME(data)>& strahlkorper_a,     \
      const ylm::Strahlkorper<FRAME(data)>& strahlkorper_b);
GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Grid, Frame::Distorted, Frame::Inertial))
#undef INSTANTIATE
#undef FRAME
