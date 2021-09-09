// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/SizeOfElement.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Limiters/KxrcfTci.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Limiters/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct TestPackagedData {
  Variables<tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD,
                       grmhd::ValenciaDivClean::Tags::TildeTau,
                       grmhd::ValenciaDivClean::Tags::TildeS<>>>
      volume_data;
  Mesh<3> mesh;
};

DirectionMap<3, DataVector> make_dirmap_of_datavectors_from_value(
    const size_t size, const double value) noexcept {
  DirectionMap<3, DataVector> result{};
  for (const auto& dir : Direction<3>::all_directions()) {
    result[dir] = DataVector(size, value);
  }
  return result;
}

void test_kxrcf_work(
    const bool expected_detection, const double kxrcf_constant,
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3>& tilde_s, const Mesh<3>& mesh,
    const Element<3>& element,
    const ElementMap<3, Frame::Inertial>& element_map,
    const DirectionMap<3, DataVector>& neighbor_densities,
    const DirectionMap<3, DataVector>& neighbor_energies) noexcept {
  // Check that this help function is called correctly
  ASSERT(element.neighbors().size() == neighbor_densities.size(),
         "The test helper was passed inconsistent data.");
  ASSERT(element.neighbors().size() == neighbor_energies.size(),
         "The test helper was passed inconsistent data.");
  for (const auto& dir : Direction<3>::all_directions()) {
    const auto& ids = element.neighbors().at(dir).ids();
    ASSERT(ids.size() == 1,
           "The test helper test_kxrcf_work isn't set up for h-refinement.");
  }

  // Create and fill neighbor data
  std::unordered_map<std::pair<Direction<3>, ElementId<3>>, TestPackagedData,
                     boost::hash<std::pair<Direction<3>, ElementId<3>>>>
      neighbor_data{};
  for (const auto& dir : Direction<3>::all_directions()) {
    const auto& id = *(element.neighbors().at(dir).ids().begin());
    TestPackagedData& neighbor = neighbor_data[std::make_pair(dir, id)];
    neighbor.mesh = mesh;
    neighbor.volume_data.initialize(mesh.number_of_grid_points(), 0.);
    get(get<grmhd::ValenciaDivClean::Tags::TildeD>(neighbor.volume_data)) =
        neighbor_densities.at(dir);
    get(get<grmhd::ValenciaDivClean::Tags::TildeTau>(neighbor.volume_data)) =
        neighbor_energies.at(dir);
  }

  const auto element_size = size_of_element(element_map);
  const Scalar<DataVector> det_jacobian{
      {1. /
       get(determinant(element_map.inv_jacobian(logical_coordinates(mesh))))}};

  // Create and fill unit normals
  DirectionMap<3, std::optional<Variables<
                      tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                 evolution::dg::Tags::NormalCovector<3>>>>>
      normals_and_magnitudes{};
  for (const auto& dir : Direction<3>::all_directions()) {
    const auto boundary_mesh = mesh.slice_away(dir.dimension());
    normals_and_magnitudes[dir] =
        Variables<tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                             evolution::dg::Tags::NormalCovector<3>>>(
            boundary_mesh.number_of_grid_points());
    auto& covector = get<evolution::dg::Tags::NormalCovector<3>>(
        normals_and_magnitudes[dir].value());
    auto& normal_magnitude = get<evolution::dg::Tags::MagnitudeOfNormal>(
        normals_and_magnitudes[dir].value());
    unnormalized_face_normal(make_not_null(&(covector)),
                             mesh.slice_away(dir.dimension()), element_map,
                             dir);
    normal_magnitude = magnitude(covector);
    for (size_t d = 0; d < 3; ++d) {
      covector.get(d) /= get(normal_magnitude);
    }
  }

  const bool tci_detection =
      grmhd::ValenciaDivClean::Limiters::Tci::kxrcf_indicator(
          kxrcf_constant, tilde_d, tilde_tau, tilde_s, mesh, element,
          element_size, det_jacobian, normals_and_magnitudes, neighbor_data);
  CHECK(tci_detection == expected_detection);
}

void test_kxrcf() noexcept {
  const auto element = TestHelpers::Limiters::make_element<3>();
  const Mesh<3> mesh(3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const Affine xi_map{-1., 1., 3., 4.2};
  const Affine eta_map{-1., 1., -3., -2.4};
  const Affine zeta_map{-1., 1., 3., 7.};
  const auto coordmap =
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Affine3D(xi_map, eta_map, zeta_map));
  const ElementMap<3, Frame::Inertial> element_map(element.id(),
                                                   coordmap->get_clone());

  const DataVector zero(mesh.number_of_grid_points(), 0.);

  Scalar<DataVector> tilde_d{
      // clang-format off
      DataVector{{0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0.4, 0.3, 0.2, 0.3, 0.2, 0.2, 0.2, 0.2, 0.1}}};
  // clang-format on
  tnsr::i<DataVector, 3> tilde_s{};
  get<0>(tilde_s) = zero;
  get<1>(tilde_s) = zero;
  get<2>(tilde_s) =
      // clang-format off
      DataVector{{0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  -0.1, -0.4, -0.2, -0.8, -0.3, -0.6, -1.2, -1.8, -0.9}};
  // clang-format on
  Scalar<DataVector> tilde_tau{
      // clang-format off
      DataVector{{1.1, 1.3, 0.8, 1.3, 0.7, 0.6, 0.9, 1.2, 1.1,
                  0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0.}}};
  // clang-format on

  const auto neighbor_densities =
      make_dirmap_of_datavectors_from_value(mesh.number_of_grid_points(), 0.);
  const auto neighbor_energies =
      make_dirmap_of_datavectors_from_value(mesh.number_of_grid_points(), 0.);

  // In realistic uses of the TCI, the kxrcf_constant would be set to a finite
  // value, perhaps 1, and different solutions may or may not trigger limiting.
  // For ease of testing, we do the opposite: we fix the solution and vary the
  // threshold.

  // trigger: density inflow at upper zeta boundary
  double kxrcf_constant = 0.;
  test_kxrcf_work(true, kxrcf_constant, tilde_d, tilde_tau, tilde_s, mesh,
                  element, element_map, neighbor_densities, neighbor_energies);

  // trigger: density inflow on subset of upper zeta boundary
  get<2>(tilde_s) =
      // clang-format off
      DataVector{{0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0.8, -0.4, -0.2, 0.1, 0.3, -0.6, -1.2, -1.8, -0.9}};
  // clang-format on
  test_kxrcf_work(true, kxrcf_constant, tilde_d, tilde_tau, tilde_s, mesh,
                  element, element_map, neighbor_densities, neighbor_energies);

  // trigger: threshold is just met
  kxrcf_constant = 1.26;
  test_kxrcf_work(true, kxrcf_constant, tilde_d, tilde_tau, tilde_s, mesh,
                  element, element_map, neighbor_densities, neighbor_energies);

  // no trigger: threshold is not met
  kxrcf_constant = 1.27;
  test_kxrcf_work(false, kxrcf_constant, tilde_d, tilde_tau, tilde_s, mesh,
                  element, element_map, neighbor_densities, neighbor_energies);

  // trigger: same threshold, but now lower eta boundary also contributes
  get(tilde_d) =
      // clang-format off
      DataVector{{0., 0.2, 0.2, 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0.4, 0.3, 0.2, 0.3, 0.2, 0.2, 0.2, 0.2, 0.1}};
  // clang-format on
  get<1>(tilde_s) =
      // clang-format off
      DataVector{{0., 0.2, 0.2, 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0.1, 0.4, 0.2, 0., 0., 0., 0., 0., 0.}};
  // clang-format on
  test_kxrcf_work(true, kxrcf_constant, tilde_d, tilde_tau, tilde_s, mesh,
                  element, element_map, neighbor_densities, neighbor_energies);

  // trigger: energy inflow at lower zeta boundary
  kxrcf_constant = 0.;
  get(tilde_d) =
      // clang-format off
      DataVector{{0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0.4, 0.3, 0.2, 0.3, 0.2, 0.2, 0.2, 0.2, 0.1}};
  // clang-format on
  get<1>(tilde_s) = zero;
  get<2>(tilde_s) =
      // clang-format off
      DataVector{{0.3, 0.1, 0.2, 0.4, 0.2, 0.3, 0.6, 0.4, 0.7,
                  0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0.}};
  // clang-format on
  test_kxrcf_work(true, kxrcf_constant, tilde_d, tilde_tau, tilde_s, mesh,
                  element, element_map, neighbor_densities, neighbor_energies);

  // no trigger: all boundaries are outflow
  get<0>(tilde_s) =
      // clang-format off
      DataVector{{-0.2, 0., 0.05, -0.2, 0., 0.3, -0.8, 0., 0.2,
                  -0.4, 0., 0.07, -0.9, 0., 0.7, -0.3, 0., 0.8,
                  -0.7, 0., 0.1, -1.3, 0., 0.8, -0.2, 0., 0.7}};
  // clang-format on
  get<1>(tilde_s) =
      // clang-format off
      DataVector{{-0.2, -0.1, -0.05, 0., 0., 0., 0.3, 1.2, 2.3,
                  -0.4, -0.1, -0.1, 0., 0., 0., 0.3, 0.8, 1.6,
                  -0.7, -0.2, -0.1, 0., 0., 0., 0.2, 0.2, 0.7}};
  // clang-format on
  get<2>(tilde_s) =
      // clang-format off
      DataVector{{-0.2, -0.1, -0.05, -0.1, -0.1, -0.4, -0.3, -1.2, -2.3,
                  0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0.7, 1.3, 1.1, 0.5, 1., 0.2, 0.2, 0.3, 0.3}};
  // clang-format on
  test_kxrcf_work(false, kxrcf_constant, tilde_d, tilde_tau, tilde_s, mesh,
                  element, element_map, neighbor_densities, neighbor_energies);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.Limiters.KxrcfTci",
                  "[Unit][Evolution]") {
  test_kxrcf();
}
