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
#include "Evolution/Systems/NewtonianEuler/Limiters/KxrcfTci.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Limiters/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <size_t VolumeDim>
struct TestPackagedData {
  Variables<tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                       NewtonianEuler::Tags::MomentumDensity<VolumeDim>,
                       NewtonianEuler::Tags::EnergyDensity>>
      volume_data;
  Mesh<VolumeDim> mesh;
};

template <size_t VolumeDim>
DirectionMap<VolumeDim, DataVector> make_dirmap_of_datavectors_from_value(
    const size_t size, const double value) {
  DirectionMap<VolumeDim, DataVector> result{};
  for (const auto& dir : Direction<VolumeDim>::all_directions()) {
    result[dir] = DataVector(size, value);
  }
  return result;
}

template <size_t VolumeDim>
void test_kxrcf_work(
    const bool expected_detection, const double kxrcf_constant,
    const Scalar<DataVector>& cons_density,
    const tnsr::I<DataVector, VolumeDim>& cons_momentum,
    const Scalar<DataVector>& cons_energy, const Mesh<VolumeDim>& mesh,
    const Element<VolumeDim>& element,
    const ElementMap<VolumeDim, Frame::Inertial>& element_map,
    const DirectionMap<VolumeDim, DataVector>& neighbor_densities,
    const DirectionMap<VolumeDim, DataVector>& neighbor_energies) {
  // Check that this help function is called correctly
  ASSERT(element.neighbors().size() == neighbor_densities.size(),
         "The test helper was passed inconsistent data.");
  ASSERT(element.neighbors().size() == neighbor_energies.size(),
         "The test helper was passed inconsistent data.");
  for (const auto& dir : Direction<VolumeDim>::all_directions()) {
    const auto& ids = element.neighbors().at(dir).ids();
    ASSERT(ids.size() == 1,
           "The test helper test_kxrcf_work isn't set up for h-refinement.");
  }

  // Create and fill neighbor data
  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
      TestPackagedData<VolumeDim>,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      neighbor_data{};
  for (const auto& dir : Direction<VolumeDim>::all_directions()) {
    const auto& id = *(element.neighbors().at(dir).ids().begin());
    TestPackagedData<VolumeDim>& neighbor =
        neighbor_data[std::make_pair(dir, id)];
    neighbor.mesh = mesh;
    neighbor.volume_data.initialize(mesh.number_of_grid_points(), 0.);
    get(get<NewtonianEuler::Tags::MassDensityCons>(neighbor.volume_data)) =
        neighbor_densities.at(dir);
    get(get<NewtonianEuler::Tags::EnergyDensity>(neighbor.volume_data)) =
        neighbor_energies.at(dir);
  }

  const auto element_size = size_of_element(element_map);
  const Scalar<DataVector> det_jacobian{
      {1. /
       get(determinant(element_map.inv_jacobian(logical_coordinates(mesh))))}};

  // Create and fill unit normals
  DirectionMap<VolumeDim, std::optional<Variables<tmpl::list<
                              evolution::dg::Tags::MagnitudeOfNormal,
                              evolution::dg::Tags::NormalCovector<VolumeDim>>>>>
      normals_and_magnitudes{};
  for (const auto& dir : Direction<VolumeDim>::all_directions()) {
    const auto boundary_mesh = mesh.slice_away(dir.dimension());
    normals_and_magnitudes[dir] =
        Variables<tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                             evolution::dg::Tags::NormalCovector<VolumeDim>>>(
            boundary_mesh.number_of_grid_points());
    auto& covector = get<evolution::dg::Tags::NormalCovector<VolumeDim>>(
        normals_and_magnitudes[dir].value());
    auto& normal_magnitude = get<evolution::dg::Tags::MagnitudeOfNormal>(
        normals_and_magnitudes[dir].value());
    unnormalized_face_normal(make_not_null(&(covector)),
                             mesh.slice_away(dir.dimension()), element_map,
                             dir);
    normal_magnitude = magnitude(covector);
    for (size_t d = 0; d < VolumeDim; ++d) {
      covector.get(d) /= get(normal_magnitude);
    }
  }

  const bool tci_detection = NewtonianEuler::Limiters::Tci::kxrcf_indicator(
      kxrcf_constant, cons_density, cons_momentum, cons_energy, mesh, element,
      element_size, det_jacobian, normals_and_magnitudes, neighbor_data);
  CHECK(tci_detection == expected_detection);
}

void test_kxrcf_1d() {
  const auto element = TestHelpers::Limiters::make_element<1>();
  const Mesh<1> mesh(3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  using Affine = domain::CoordinateMaps::Affine;
  const Affine xi_map{-1., 1., 3., 4.2};
  const auto coordmap =
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          xi_map);
  const ElementMap<1, Frame::Inertial> element_map(element.id(),
                                                   coordmap->get_clone());

  Scalar<DataVector> cons_density{DataVector{{0.4, 0.2, 0.}}};
  tnsr::I<DataVector, 1> cons_momentum{DataVector{{0.2, 0.1, 0.05}}};
  Scalar<DataVector> cons_energy{DataVector{{0., 1.3, 0.}}};

  const auto neighbor_densities = make_dirmap_of_datavectors_from_value<1>(
      mesh.number_of_grid_points(), 0.);
  const auto neighbor_energies = make_dirmap_of_datavectors_from_value<1>(
      mesh.number_of_grid_points(), 0.);

  // In realistic uses of the TCI, the kxrcf_constant would be set to a finite
  // value, perhaps 1, and different solutions may or may not trigger limiting.
  // For ease of testing, we do the opposite: we fix the solution and vary the
  // threshold.

  // trigger: density inflow at lower xi boundary
  double kxrcf_constant = 0.;
  test_kxrcf_work(true, kxrcf_constant, cons_density, cons_momentum,
                  cons_energy, mesh, element, element_map, neighbor_densities,
                  neighbor_energies);

  // trigger: threshold is just met
  kxrcf_constant = 4.81;
  test_kxrcf_work(true, kxrcf_constant, cons_density, cons_momentum,
                  cons_energy, mesh, element, element_map, neighbor_densities,
                  neighbor_energies);

  // no trigger: threshold is not met
  kxrcf_constant = 4.82;
  test_kxrcf_work(false, kxrcf_constant, cons_density, cons_momentum,
                  cons_energy, mesh, element, element_map, neighbor_densities,
                  neighbor_energies);

  // no trigger: density has no jump at lower xi boundary
  kxrcf_constant = 0.;
  get(cons_density)[0] = 0.;
  test_kxrcf_work(false, kxrcf_constant, cons_density, cons_momentum,
                  cons_energy, mesh, element, element_map, neighbor_densities,
                  neighbor_energies);

  // trigger: energy inflow at upper xi boundary
  cons_density = Scalar<DataVector>{DataVector{{0., 0.3, 0.}}};
  cons_momentum = tnsr::I<DataVector, 1>{DataVector{{-0.2, 0.1, -0.05}}};
  cons_energy = Scalar<DataVector>{DataVector{{0., 1.3, 1.7}}};
  test_kxrcf_work(true, kxrcf_constant, cons_density, cons_momentum,
                  cons_energy, mesh, element, element_map, neighbor_densities,
                  neighbor_energies);

  // no trigger: energy has no jump at upper xi boundary
  get(cons_energy)[mesh.number_of_grid_points() - 1] = 0.;
  test_kxrcf_work(false, kxrcf_constant, cons_density, cons_momentum,
                  cons_energy, mesh, element, element_map, neighbor_densities,
                  neighbor_energies);

  // no trigger: all boundaries are outflow
  cons_density = Scalar<DataVector>{DataVector{{0.4, 0.3, 0.2}}};
  cons_momentum = tnsr::I<DataVector, 1>{DataVector{{-0.2, 0.1, 0.05}}};
  cons_energy = Scalar<DataVector>{DataVector{{1.4, 1.3, 1.7}}};
  test_kxrcf_work(false, kxrcf_constant, cons_density, cons_momentum,
                  cons_energy, mesh, element, element_map, neighbor_densities,
                  neighbor_energies);
}

void test_kxrcf_2d() {
  const auto element = TestHelpers::Limiters::make_element<2>();
  const Mesh<2> mesh(3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  using Affine = domain::CoordinateMaps::Affine;
  using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  const Affine xi_map{-1., 1., 3., 4.2};
  const Affine eta_map{-1., 1., 3., 7.};
  const auto coordmap =
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          Affine2D(xi_map, eta_map));
  const ElementMap<2, Frame::Inertial> element_map(element.id(),
                                                   coordmap->get_clone());

  const DataVector zero(mesh.number_of_grid_points(), 0.);

  Scalar<DataVector> cons_density{
      DataVector{{0.4, 0.3, 0.2, 0., 0., 0., 0., 0., 0.}}};
  tnsr::I<DataVector, 2> cons_momentum{};
  get<0>(cons_momentum) = zero;
  get<1>(cons_momentum) = DataVector{{0.2, 0.1, 0.05, 0., 0., 0., 0., 0., 0.}};
  Scalar<DataVector> cons_energy{
      DataVector{{0., 0., 0., 0., 0., 0., 0.3, 0.2, 0.1}}};

  const auto neighbor_densities = make_dirmap_of_datavectors_from_value<2>(
      mesh.number_of_grid_points(), 0.);
  const auto neighbor_energies = make_dirmap_of_datavectors_from_value<2>(
      mesh.number_of_grid_points(), 0.);

  // In realistic uses of the TCI, the kxrcf_constant would be set to a finite
  // value, perhaps 1, and different solutions may or may not trigger limiting.
  // For ease of testing, we do the opposite: we fix the solution and vary the
  // threshold.

  // trigger: density inflow at lower eta boundary
  double kxrcf_constant = 0.;
  test_kxrcf_work(true, kxrcf_constant, cons_density, cons_momentum,
                  cons_energy, mesh, element, element_map, neighbor_densities,
                  neighbor_energies);

  // trigger: density inflow on subset of lower eta boundary
  get<1>(cons_momentum) = DataVector{{0.2, 0.1, -0.05, 0., 0., 0., 0., 0., 0.}};
  test_kxrcf_work(true, kxrcf_constant, cons_density, cons_momentum,
                  cons_energy, mesh, element, element_map, neighbor_densities,
                  neighbor_energies);

  // trigger: threshold is just met
  kxrcf_constant = 0.776;
  test_kxrcf_work(true, kxrcf_constant, cons_density, cons_momentum,
                  cons_energy, mesh, element, element_map, neighbor_densities,
                  neighbor_energies);

  // no trigger: threshold is not met
  kxrcf_constant = 0.777;
  test_kxrcf_work(false, kxrcf_constant, cons_density, cons_momentum,
                  cons_energy, mesh, element, element_map, neighbor_densities,
                  neighbor_energies);

  // trigger: same threshold, but now upper xi boundary also contributes
  get(cons_density) = DataVector{{0.4, 0.3, 0.2, 0., 0., 0., 0., 0., 0.8}};
  get<0>(cons_momentum) = DataVector{{0., 0., 0., 0., 0., 0., 0., 0., -1.4}};
  test_kxrcf_work(true, kxrcf_constant, cons_density, cons_momentum,
                  cons_energy, mesh, element, element_map, neighbor_densities,
                  neighbor_energies);

  // trigger: energy inflow at upper eta boundary
  kxrcf_constant = 0.;
  get(cons_density) = DataVector{{0.4, 0.3, 0.2, 0., 0., 0., 0., 0., 0.}};
  get<0>(cons_momentum) = zero;
  get<1>(cons_momentum) =
      DataVector{{0., 0., 0., 0., 0., 0., -0.8, -1.3, -1.2}};
  test_kxrcf_work(true, kxrcf_constant, cons_density, cons_momentum,
                  cons_energy, mesh, element, element_map, neighbor_densities,
                  neighbor_energies);

  // no trigger: all boundaries are outflow
  get<0>(cons_momentum) =
      DataVector{{-0.2, 0., 0.05, -1.2, 0., 0.3, -0.9, 0., 0.4}};
  get<1>(cons_momentum) =
      DataVector{{-0.2, -0.1, -0.05, 0., 0., 0., 0.3, 1.2, 2.3}};
  test_kxrcf_work(false, kxrcf_constant, cons_density, cons_momentum,
                  cons_energy, mesh, element, element_map, neighbor_densities,
                  neighbor_energies);
}

void test_kxrcf_3d() {
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
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          Affine3D(xi_map, eta_map, zeta_map));
  const ElementMap<3, Frame::Inertial> element_map(element.id(),
                                                   coordmap->get_clone());

  const DataVector zero(mesh.number_of_grid_points(), 0.);

  Scalar<DataVector> cons_density{
      // clang-format off
      DataVector{{0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0.4, 0.3, 0.2, 0.3, 0.2, 0.2, 0.2, 0.2, 0.1}}};
  // clang-format on
  tnsr::I<DataVector, 3> cons_momentum{};
  get<0>(cons_momentum) = zero;
  get<1>(cons_momentum) = zero;
  get<2>(cons_momentum) =
      // clang-format off
      DataVector{{0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  -0.1, -0.4, -0.2, -0.8, -0.3, -0.6, -1.2, -1.8, -0.9}};
  // clang-format on
  Scalar<DataVector> cons_energy{
      // clang-format off
      DataVector{{1.1, 1.3, 0.8, 1.3, 0.7, 0.6, 0.9, 1.2, 1.1,
                  0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0.}}};
  // clang-format on

  const auto neighbor_densities = make_dirmap_of_datavectors_from_value<3>(
      mesh.number_of_grid_points(), 0.);
  const auto neighbor_energies = make_dirmap_of_datavectors_from_value<3>(
      mesh.number_of_grid_points(), 0.);

  // In realistic uses of the TCI, the kxrcf_constant would be set to a finite
  // value, perhaps 1, and different solutions may or may not trigger limiting.
  // For ease of testing, we do the opposite: we fix the solution and vary the
  // threshold.

  // trigger: density inflow at upper zeta boundary
  double kxrcf_constant = 0.;
  test_kxrcf_work(true, kxrcf_constant, cons_density, cons_momentum,
                  cons_energy, mesh, element, element_map, neighbor_densities,
                  neighbor_energies);

  // trigger: density inflow on subset of upper zeta boundary
  get<2>(cons_momentum) =
      // clang-format off
      DataVector{{0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0.8, -0.4, -0.2, 0.1, 0.3, -0.6, -1.2, -1.8, -0.9}};
  // clang-format on
  test_kxrcf_work(true, kxrcf_constant, cons_density, cons_momentum,
                  cons_energy, mesh, element, element_map, neighbor_densities,
                  neighbor_energies);

  // trigger: threshold is just met
  kxrcf_constant = 1.26;
  test_kxrcf_work(true, kxrcf_constant, cons_density, cons_momentum,
                  cons_energy, mesh, element, element_map, neighbor_densities,
                  neighbor_energies);

  // no trigger: threshold is not met
  kxrcf_constant = 1.27;
  test_kxrcf_work(false, kxrcf_constant, cons_density, cons_momentum,
                  cons_energy, mesh, element, element_map, neighbor_densities,
                  neighbor_energies);

  // trigger: same threshold, but now lower eta boundary also contributes
  get(cons_density) =
      // clang-format off
      DataVector{{0., 0.2, 0.2, 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0.4, 0.3, 0.2, 0.3, 0.2, 0.2, 0.2, 0.2, 0.1}};
  // clang-format on
  get<1>(cons_momentum) =
      // clang-format off
      DataVector{{0., 0.2, 0.2, 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0.1, 0.4, 0.2, 0., 0., 0., 0., 0., 0.}};
  // clang-format on
  test_kxrcf_work(true, kxrcf_constant, cons_density, cons_momentum,
                  cons_energy, mesh, element, element_map, neighbor_densities,
                  neighbor_energies);

  // trigger: energy inflow at lower zeta boundary
  kxrcf_constant = 0.;
  get(cons_density) =
      // clang-format off
      DataVector{{0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0.4, 0.3, 0.2, 0.3, 0.2, 0.2, 0.2, 0.2, 0.1}};
  // clang-format on
  get<1>(cons_momentum) = zero;
  get<2>(cons_momentum) =
      // clang-format off
      DataVector{{0.3, 0.1, 0.2, 0.4, 0.2, 0.3, 0.6, 0.4, 0.7,
                  0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0.}};
  // clang-format on
  test_kxrcf_work(true, kxrcf_constant, cons_density, cons_momentum,
                  cons_energy, mesh, element, element_map, neighbor_densities,
                  neighbor_energies);

  // no trigger: all boundaries are outflow
  get<0>(cons_momentum) =
      // clang-format off
      DataVector{{-0.2, 0., 0.05, -0.2, 0., 0.3, -0.8, 0., 0.2,
                  -0.4, 0., 0.07, -0.9, 0., 0.7, -0.3, 0., 0.8,
                  -0.7, 0., 0.1, -1.3, 0., 0.8, -0.2, 0., 0.7}};
  // clang-format on
  get<1>(cons_momentum) =
      // clang-format off
      DataVector{{-0.2, -0.1, -0.05, 0., 0., 0., 0.3, 1.2, 2.3,
                  -0.4, -0.1, -0.1, 0., 0., 0., 0.3, 0.8, 1.6,
                  -0.7, -0.2, -0.1, 0., 0., 0., 0.2, 0.2, 0.7}};
  // clang-format on
  get<2>(cons_momentum) =
      // clang-format off
      DataVector{{-0.2, -0.1, -0.05, -0.1, -0.1, -0.4, -0.3, -1.2, -2.3,
                  0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0.7, 1.3, 1.1, 0.5, 1., 0.2, 0.2, 0.3, 0.3}};
  // clang-format on
  test_kxrcf_work(false, kxrcf_constant, cons_density, cons_momentum,
                  cons_energy, mesh, element, element_map, neighbor_densities,
                  neighbor_energies);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Limiters.KxrcfTci",
                  "[Unit][Evolution]") {
  test_kxrcf_1d();
  test_kxrcf_2d();
  test_kxrcf_3d();
}
