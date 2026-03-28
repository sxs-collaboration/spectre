// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>

#include "Domain/Creators/NonconformingSphericalShells.hpp"
#include "Domain/Domain.hpp"
#include "Evolution/DiscontinuousGalerkin/InterfaceDataPolicy.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarInfo.hpp"
#include "Evolution/DiscontinuousGalerkin/TimeSteppingPolicy.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarInterpolator.hpp"
#include "NumericalAlgorithms/Spectral/SegmentSize.hpp"
#include "Utilities/MakeArray.hpp"

namespace evolution::dg {
namespace {
template <size_t Dim>
void test(
    const std::array<Spectral::SegmentSize, Dim - 1>& mortar_size,
    const std::optional<::dg::MortarInterpolator<Dim>>& mortar_interpolator) {
  const MortarInfo<Dim> nonconforming_mortar_info{
      {.interpolator = mortar_interpolator,
       .interface_data_policy =
           InterfaceDataPolicy::NonconformingSelfInterpolates,
       .time_stepping_policy = TimeSteppingPolicy::EqualRate}};
  CHECK(nonconforming_mortar_info.interface_data_policy() ==
        InterfaceDataPolicy::NonconformingSelfInterpolates);
  CHECK(nonconforming_mortar_info.time_stepping_policy() ==
        TimeSteppingPolicy::EqualRate);
  CHECK(nonconforming_mortar_info.mortar_size() ==
        make_array<Dim - 1>(Spectral::SegmentSize::Uninitialized));
  CHECK(nonconforming_mortar_info.interpolator() == mortar_interpolator);
  const auto deserialized_nonconforming_mortar_info =
      serialize_and_deserialize(nonconforming_mortar_info);
  CHECK(nonconforming_mortar_info == deserialized_nonconforming_mortar_info);
  const MortarInfo<Dim> conforming_mortar_info{
      {.mortar_size = mortar_size,
       .interface_data_policy = InterfaceDataPolicy::CopyProject,
       .time_stepping_policy = TimeSteppingPolicy::Conservative}};
  CHECK(conforming_mortar_info.interface_data_policy() ==
        InterfaceDataPolicy::CopyProject);
  CHECK(conforming_mortar_info.time_stepping_policy() ==
        TimeSteppingPolicy::Conservative);
  CHECK(conforming_mortar_info.mortar_size() == mortar_size);
  CHECK(conforming_mortar_info.interpolator() == std::nullopt);
  const auto deserialized_conforming_mortar_info =
      serialize_and_deserialize(conforming_mortar_info);
  CHECK(conforming_mortar_info == deserialized_conforming_mortar_info);
  CHECK(conforming_mortar_info != nonconforming_mortar_info);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.MortarInfo", "[Unit][Evolution]") {
  test<1>({{}}, std::nullopt);
  test<2>({{Spectral::SegmentSize::Full}}, std::nullopt);
  const auto creator = domain::creators::NonconformingSphericalShells(
      2.0, 3.0, 4.0, 0, 0, 5, 8, 11, nullptr, nullptr);
  const auto domain = creator.domain();
  test<3>(
      {{Spectral::SegmentSize::UpperHalf, Spectral::SegmentSize::LowerHalf}},
      ::dg::MortarInterpolator{
          ElementId<3>{0},
          DirectionalId<3>{Direction<3>::upper_zeta(), ElementId<3>{6}}, domain,
          Mesh<2>{11_st, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto},
          Mesh<2>{std::array{8_st, 15_st},
                  std::array{Spectral::Basis::SphericalHarmonic,
                             Spectral::Basis::SphericalHarmonic},
                  std::array{Spectral::Quadrature::Gauss,
                             Spectral::Quadrature::Equiangular}}});
}
}  // namespace evolution::dg
