// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "Evolution/DiscontinuousGalerkin/InterfaceDataPolicy.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarInfo.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/SegmentSize.hpp"
#include "Utilities/MakeArray.hpp"

namespace evolution::dg {
namespace {
template <size_t Dim>
void test(const std::array<Spectral::SegmentSize, Dim - 1>& mortar_size) {
  const MortarInfo<Dim> nonconforming_mortar_info{
      {.policy = InterfaceDataPolicy::NonconformingSelfInterpolates}};
  CHECK(nonconforming_mortar_info.policy() ==
        InterfaceDataPolicy::NonconformingSelfInterpolates);
  CHECK(nonconforming_mortar_info.mortar_size() ==
        make_array<Dim - 1>(Spectral::SegmentSize::Uninitialized));
  const auto deserialized_nonconforming_mortar_info =
      serialize_and_deserialize(nonconforming_mortar_info);
  CHECK(nonconforming_mortar_info == deserialized_nonconforming_mortar_info);
  const MortarInfo<Dim> conforming_mortar_info{
      {.mortar_size = mortar_size, .policy = InterfaceDataPolicy::CopyProject}};
  CHECK(conforming_mortar_info.policy() == InterfaceDataPolicy::CopyProject);
  CHECK(conforming_mortar_info.mortar_size() == mortar_size);
  const auto deserialized_conforming_mortar_info =
      serialize_and_deserialize(conforming_mortar_info);
  CHECK(conforming_mortar_info == deserialized_conforming_mortar_info);
  CHECK(conforming_mortar_info != nonconforming_mortar_info);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.MortarInfo", "[Unit][Evolution]") {
  test<1>({{}});
  test<2>({{Spectral::SegmentSize::Full}});
  test<3>(
      {{Spectral::SegmentSize::UpperHalf, Spectral::SegmentSize::LowerHalf}});
}
}  // namespace evolution::dg
