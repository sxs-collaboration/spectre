// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace StrahlkorperTags {

template <typename Frame>
void PhysicalCenterCompute<Frame>::function(
    const gsl::not_null<std::array<double, 3>*> physical_center,
    const ::Strahlkorper<Frame>& strahlkorper) {
  *physical_center = strahlkorper.physical_center();
}

}  // namespace StrahlkorperTags

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data) \
  template struct StrahlkorperTags::PhysicalCenterCompute<FRAME(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATE, (Frame::Grid, Frame::Inertial))
#undef INSTANTIATE
#undef FRAME
