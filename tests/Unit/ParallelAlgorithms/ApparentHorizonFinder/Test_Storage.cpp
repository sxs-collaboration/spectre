// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <deque>
#include <optional>
#include <set>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Destination.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"

namespace ah {
namespace {
// The only thing to test with the storage is the serialization since it's just
// structs with public members
template <typename Fr>
void test_storage() {
  const ah::Storage::VolumeVariables<Fr> volume_variables{
      Mesh<3>{3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto},
      Variables<ah::vars_to_interpolate_to_target<3, Fr>>{4, 4.321}};
  test_serialization(volume_variables);

  ah::Storage::Iteration<Fr> iteration{
      ylm::Strahlkorper<Fr>{4_st, 3.0, std::array{0.0, 0.1, 0.2}},
      std::optional<std::vector<BlockLogicalCoords<3>>>{{std::nullopt}},
      Variables<ah::vars_to_interpolate_to_target<3, Fr>>{6, 9.876},
      std::vector<bool>{false, true, false, false, true, true},
      {},
      {},
      {},
      {2}};
  test_serialization(iteration);
  CHECK_FALSE(iteration.interpolation_is_complete());
  for (size_t i = 0; i < iteration.indices_interpolated_to_thus_far.size();
       ++i) {
    iteration.indices_interpolated_to_thus_far[i] = true;
  }
  CHECK(iteration.interpolation_is_complete());

  const ah::Storage::SingleTimeStorage<Fr> single_time_storage{
      std::unordered_map<ElementId<3>, ah::Storage::VolumeVariables<Fr>>{
          {ElementId<3>{0}, volume_variables}},
      {},
      iteration,
      iteration.strahlkorper,
      Destination::ControlSystem};
  test_serialization(single_time_storage);

  const ah::Storage::PreviousSurface<Fr> previous_surface{
      LinkedMessageId<double>{3.0, {2.0}}, iteration.strahlkorper};
  test_serialization(previous_surface);

  // Check we can use PreviousSurface with `emplace`
  std::deque<ah::Storage::PreviousSurface<Fr>> previous_surfaces{};
  previous_surfaces.emplace_front(LinkedMessageId<double>{1.0, std::nullopt},
                                  iteration.strahlkorper);
  CHECK(previous_surfaces.front().time ==
        LinkedMessageId<double>{1.0, std::nullopt});
  CHECK(previous_surfaces.front().surface == iteration.strahlkorper);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.Storage",
                  "[ApparentHorizonFinder][Unit]") {
  test_storage<::Frame::Grid>();
  test_storage<::Frame::Distorted>();
  test_storage<::Frame::Inertial>();
}
}  // namespace ah
