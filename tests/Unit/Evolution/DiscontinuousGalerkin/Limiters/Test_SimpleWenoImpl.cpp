// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"  // IWYU pragma: keep
#include "Domain/ElementId.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Side.hpp"
#include "ErrorHandling/Error.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/SimpleWenoImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoOscillationIndicator.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Limiters/TestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_include "Domain/OrientationMapHelpers.hpp"
// IWYU pragma: no_forward_declare Tags::Mean
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

namespace {

template <size_t VolumeDim>
struct VectorTag : db::SimpleTag {
  using type = tnsr::I<DataVector, VolumeDim>;
};

template <size_t VolumeDim, typename Tag>
struct DummyPackagedData {
  Variables<tmpl::list<Tag>> volume_data;
  tuples::TaggedTuple<::Tags::Mean<Tag>> means;
  Mesh<VolumeDim> mesh;
};

template <size_t VolumeDim>
using VariablesMap = std::unordered_map<
    std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
    Variables<tmpl::list<VectorTag<VolumeDim>>>,
    boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>;

template <size_t VolumeDim>
std::unordered_map<
    std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
    DummyPackagedData<VolumeDim, VectorTag<VolumeDim>>,
    boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
make_neighbor_data_from_neighbor_vars(
    const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
    const VariablesMap<VolumeDim>& neighbor_vars) noexcept {
  const auto make_tuple_of_means = [&mesh](
      const Variables<tmpl::list<VectorTag<VolumeDim>>>&
          vars_to_average) noexcept {
    tuples::TaggedTuple<::Tags::Mean<VectorTag<VolumeDim>>> result;
    for (size_t d = 0; d < VolumeDim; ++d) {
      get<::Tags::Mean<VectorTag<VolumeDim>>>(result).get(d) =
          mean_value(get<VectorTag<VolumeDim>>(vars_to_average).get(d), mesh);
    }
    return result;
  };

  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
      DummyPackagedData<VolumeDim, VectorTag<VolumeDim>>,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      neighbor_data{};

  for (const auto& neighbor : element.neighbors()) {
    const auto dir = neighbor.first;
    const auto id = *(neighbor.second.cbegin());
    const auto dir_and_id = std::make_pair(dir, id);
    neighbor_data[dir_and_id].volume_data = neighbor_vars.at(dir_and_id);
    neighbor_data[dir_and_id].means =
        make_tuple_of_means(neighbor_vars.at(dir_and_id));
    neighbor_data[dir_and_id].mesh = mesh;
  }

  return neighbor_data;
}

template <size_t VolumeDim>
void test_simple_weno_work(
    const tnsr::I<DataVector, VolumeDim>& local_data,
    const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        DummyPackagedData<VolumeDim, VectorTag<VolumeDim>>,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data,
    const VariablesMap<VolumeDim>& expected_neighbor_modified_vars,
    Approx local_approx = approx) noexcept {
  // First run some sanity checks on the input, and make sure the test function
  // is being called in a reasonable way
  if (element.neighbors().size() != neighbor_data.size()) {
    ERROR("Different number of neighbors from element, neighbor_data");
  }
  if (neighbor_data.size() != expected_neighbor_modified_vars.size()) {
    ERROR("Different sizes for neighbor_data, expected_neighbor_modified_vars");
  }
  for (const auto& neighbor : element.neighbors()) {
    if (neighbor.second.ids().size() > 1) {
      ERROR("Too many neighbors: h-refinement is not yet supported");
    }
    const auto dir = neighbor.first;
    const auto id = *(neighbor.second.cbegin());
    const auto dir_and_id = std::make_pair(dir, id);
    if (neighbor_data.find(dir_and_id) == neighbor_data.end()) {
      ERROR("Missing neighbor_data at an internal boundary");
    }
    if (expected_neighbor_modified_vars.find(dir_and_id) ==
        expected_neighbor_modified_vars.end()) {
      ERROR("Missing expected_neighbor_modified_vars at an internal boundary");
    }
  }

  // Buffers for simple WENO implementation
  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
      intrp::RegularGrid<VolumeDim>,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      interpolator_buffer{};
  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, DataVector,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      modified_neighbor_solution_buffer{};
  for (const auto& neighbor_and_data : neighbor_data) {
    const auto& neighbor = neighbor_and_data.first;
    modified_neighbor_solution_buffer.insert(
        make_pair(neighbor, DataVector(mesh.number_of_grid_points())));
  }

  // WENO should preserve the mean, so expected means = initial means
  const auto expected_vector_means = [&local_data, &mesh ]() noexcept {
    std::array<double, VolumeDim> means{};
    for (size_t d = 0; d < VolumeDim; ++d) {
      gsl::at(means, d) = mean_value(local_data.get(d), mesh);
    }
    return means;
  }
  ();

  auto vector_to_limit = local_data;
  const double neighbor_linear_weight = 0.001;
  // The "tensor" interface is a thin wrapper around the "single component"
  // interface, so no need to test both overloads separately.
  Limiters::Weno_detail::simple_weno_impl<VectorTag<VolumeDim>>(
      make_not_null(&interpolator_buffer),
      make_not_null(&modified_neighbor_solution_buffer),
      make_not_null(&vector_to_limit), neighbor_linear_weight, mesh, element,
      neighbor_data);

  for (size_t d = 0; d < VolumeDim; ++d) {
    CHECK(mean_value(vector_to_limit.get(d), mesh) ==
          approx(gsl::at(expected_vector_means, d)));
  }

  auto expected_vector = local_data;
  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, DataVector,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      expected_neighbor_polynomials;
  for (size_t i = 0; i < VolumeDim; ++i) {
    for (auto& neighbor_and_vars : expected_neighbor_modified_vars) {
      expected_neighbor_polynomials[neighbor_and_vars.first] =
          get<VectorTag<VolumeDim>>(neighbor_and_vars.second).get(i);
    }
    Limiters::Weno_detail::reconstruct_from_weighted_sum(
        make_not_null(&(expected_vector.get(i))), mesh, neighbor_linear_weight,
        expected_neighbor_polynomials,
        Limiters::Weno_detail::DerivativeWeight::PowTwoEll);
  }
  CHECK_ITERABLE_CUSTOM_APPROX(expected_vector, vector_to_limit, local_approx);
}

void test_simple_weno_1d(const std::unordered_set<Direction<1>>&
                             directions_of_external_boundaries = {}) noexcept {
  INFO("Test simple_weno_impl in 1D");
  CAPTURE(directions_of_external_boundaries);
  const auto mesh =
      Mesh<1>(3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
  const auto element =
      TestHelpers::Limiters::make_element<1>(directions_of_external_boundaries);
  const auto logical_coords = logical_coordinates(mesh);

  // Functions to produce dummy data on each element
  const auto make_center_tensor =
      [](const tnsr::I<DataVector, 1, Frame::Logical>& coords) noexcept {
    const auto& x = get<0>(coords);
    return tnsr::I<DataVector, 1>{{{0.4 * x - 0.1 * square(x)}}};
  };
  const auto make_lower_xi_vars =
      [](const tnsr::I<DataVector, 1, Frame::Logical>& coords,
         const double offset = 0.0) noexcept {
    const auto x = get<0>(coords) + offset;
    Variables<tmpl::list<VectorTag<1>>> vars(x.size());
    get<0>(get<VectorTag<1>>(vars)) = -0.1 + 0.3 * x - 0.1 * square(x);
    return vars;
  };
  const auto make_upper_xi_vars =
      [](const tnsr::I<DataVector, 1, Frame::Logical>& coords,
         const double offset = 0.0) noexcept {
    const auto x = get<0>(coords) + offset;
    Variables<tmpl::list<VectorTag<1>>> vars(x.size());
    get<0>(get<VectorTag<1>>(vars)) = 0.6 * x - 0.3 * square(x);
    return vars;
  };

  const auto local_data = make_center_tensor(logical_coords);
  VariablesMap<1> neighbor_vars{};
  VariablesMap<1> neighbor_modified_vars{};

  const auto shift_vars_to_local_means = [&mesh, &local_data ](
      const Variables<tmpl::list<VectorTag<1>>>& input) noexcept {
    auto result = input;
    auto& v = get<VectorTag<1>>(result);
    get<0>(v) +=
        mean_value(get<0>(local_data), mesh) - mean_value(get<0>(v), mesh);
    return result;
  };

  const auto make_neighbor_vars =
      [
        &logical_coords, &directions_of_external_boundaries, &neighbor_vars,
        &neighbor_modified_vars, &shift_vars_to_local_means
      ](const std::pair<Direction<1>, ElementId<1>>& neighbor,
        const auto make_vars) noexcept {
    if (directions_of_external_boundaries.count(neighbor.first) == 0) {
      const double offset = (neighbor.first.side() == Side::Lower ? -2.0 : 2.0);
      neighbor_vars[neighbor] = make_vars(logical_coords, offset);
      neighbor_modified_vars[neighbor] =
          shift_vars_to_local_means(make_vars(logical_coords));
    }
  };

  make_neighbor_vars(std::make_pair(Direction<1>::lower_xi(), ElementId<1>(1)),
                     make_lower_xi_vars);

  make_neighbor_vars(std::make_pair(Direction<1>::upper_xi(), ElementId<1>(2)),
                     make_upper_xi_vars);

  const auto neighbor_data =
      make_neighbor_data_from_neighbor_vars(mesh, element, neighbor_vars);

  test_simple_weno_work<1>(local_data, mesh, element, neighbor_data,
                           neighbor_modified_vars);
}

void test_simple_weno_2d(const std::unordered_set<Direction<2>>&
                             directions_of_external_boundaries = {}) noexcept {
  INFO("Test simple_weno_impl in 2D");
  CAPTURE(directions_of_external_boundaries);
  const auto mesh =
      Mesh<2>(3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
  const auto element =
      TestHelpers::Limiters::make_element<2>(directions_of_external_boundaries);
  const auto logical_coords = logical_coordinates(mesh);

  const auto make_center_tensor =
      [](const tnsr::I<DataVector, 2, Frame::Logical>& coords) noexcept {
    const auto& x = get<0>(coords);
    const auto& y = get<1>(coords);
    return tnsr::I<DataVector, 2>{
        {{x + 2.5 * y, 0.1 + 0.2 * x - 0.4 * y + 0.3 * square(x) * square(y)}}};
  };
  const auto make_lower_xi_vars =
      [](const tnsr::I<DataVector, 2, Frame::Logical>& coords,
         const double xi_offset = 0.0) noexcept {
    const auto x = get<0>(coords) + xi_offset;
    const auto& y = get<1>(coords);
    Variables<tmpl::list<VectorTag<2>>> vars(x.size());
    get<0>(get<VectorTag<2>>(vars)) = x + 2.5 * y;
    get<1>(get<VectorTag<2>>(vars)) = 3.0 + 0.2 * y;
    return vars;
  };
  const auto make_upper_xi_vars =
      [](const tnsr::I<DataVector, 2, Frame::Logical>& coords,
         const double xi_offset = 0.0) noexcept {
    const auto x = get<0>(coords) + xi_offset;
    const auto& y = get<1>(coords);
    Variables<tmpl::list<VectorTag<2>>> vars(x.size());
    get<0>(get<VectorTag<2>>(vars)) = x + 2.5 * y;
    get<1>(get<VectorTag<2>>(vars)) = -2.4 + square(x);
    return vars;
  };
  const auto make_lower_eta_vars =
      [](const tnsr::I<DataVector, 2, Frame::Logical>& coords,
         const double eta_offset = 0.0) noexcept {
    const auto& x = get<0>(coords);
    const auto y = get<1>(coords) + eta_offset;
    Variables<tmpl::list<VectorTag<2>>> vars(x.size());
    get<0>(get<VectorTag<2>>(vars)) = x + 2.5 * y;
    get<1>(get<VectorTag<2>>(vars)) = 0.2 - y;
    return vars;
  };
  const auto make_upper_eta_vars =
      [](const tnsr::I<DataVector, 2, Frame::Logical>& coords,
         const double eta_offset = 0.0) noexcept {
    const auto& x = get<0>(coords);
    const auto y = get<1>(coords) + eta_offset;
    Variables<tmpl::list<VectorTag<2>>> vars(x.size());
    get<0>(get<VectorTag<2>>(vars)) = x + 2.5 * y;
    get<1>(get<VectorTag<2>>(vars)) = 0.4 + 0.3 * x * square(y);
    return vars;
  };

  const auto local_data = make_center_tensor(logical_coords);
  VariablesMap<2> neighbor_vars{};
  VariablesMap<2> neighbor_modified_vars{};

  const auto shift_vars_to_local_means = [&mesh, &local_data ](
      const Variables<tmpl::list<VectorTag<2>>>& input) noexcept {
    auto result = input;
    auto& v = get<VectorTag<2>>(result);
    get<0>(v) +=
        mean_value(get<0>(local_data), mesh) - mean_value(get<0>(v), mesh);
    get<1>(v) +=
        mean_value(get<1>(local_data), mesh) - mean_value(get<1>(v), mesh);
    return result;
  };

  const auto make_neighbor_vars =
      [
        &logical_coords, &directions_of_external_boundaries, &neighbor_vars,
        &neighbor_modified_vars, &shift_vars_to_local_means
      ](const std::pair<Direction<2>, ElementId<2>>& neighbor,
        const auto make_vars) noexcept {
    if (directions_of_external_boundaries.count(neighbor.first) == 0) {
      const double offset = (neighbor.first.side() == Side::Lower ? -2.0 : 2.0);
      neighbor_vars[neighbor] = make_vars(logical_coords, offset);
      neighbor_modified_vars[neighbor] =
          shift_vars_to_local_means(make_vars(logical_coords));
    }
  };

  make_neighbor_vars(std::make_pair(Direction<2>::lower_xi(), ElementId<2>(1)),
                     make_lower_xi_vars);

  make_neighbor_vars(std::make_pair(Direction<2>::upper_xi(), ElementId<2>(2)),
                     make_upper_xi_vars);

  make_neighbor_vars(std::make_pair(Direction<2>::lower_eta(), ElementId<2>(3)),
                     make_lower_eta_vars);

  make_neighbor_vars(std::make_pair(Direction<2>::upper_eta(), ElementId<2>(4)),
                     make_upper_eta_vars);

  const auto neighbor_data =
      make_neighbor_data_from_neighbor_vars(mesh, element, neighbor_vars);

  test_simple_weno_work<2>(local_data, mesh, element, neighbor_data,
                           neighbor_modified_vars);
}

void test_simple_weno_3d(const std::unordered_set<Direction<3>>&
                             directions_of_external_boundaries = {}) noexcept {
  INFO("Test simple_weno_impl in 3D");
  CAPTURE(directions_of_external_boundaries);
  const auto mesh = Mesh<3>({{3, 4, 5}}, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto);
  const auto element =
      TestHelpers::Limiters::make_element<3>(directions_of_external_boundaries);
  const auto logical_coords = logical_coordinates(mesh);

  const auto make_center_tensor =
      [](const tnsr::I<DataVector, 3, Frame::Logical>& coords) noexcept {
    const auto& x = get<0>(coords);
    const auto& y = get<1>(coords);
    const auto& z = get<2>(coords);
    return tnsr::I<DataVector, 3>{
        {{0.4 * x * y * z + square(z), z, x + square(y) + cube(z)}}};
  };
  const auto make_lower_xi_vars =
      [](const tnsr::I<DataVector, 3, Frame::Logical>& coords,
         const double xi_offset = 0.0) noexcept {
    const auto x = get<0>(coords) + xi_offset;
    const auto& y = get<1>(coords);
    const auto& z = get<2>(coords);
    Variables<tmpl::list<VectorTag<3>>> vars(x.size());
    get<0>(get<VectorTag<3>>(vars)) = 0.4 * x * y * z + square(z);
    get<1>(get<VectorTag<3>>(vars)) = 0.8 * z + 0.3 * x * y;
    get<2>(get<VectorTag<3>>(vars)) = x + y;
    return vars;
  };
  const auto make_upper_xi_vars =
      [](const tnsr::I<DataVector, 3, Frame::Logical>& coords,
         const double xi_offset = 0.0) noexcept {
    const auto x = get<0>(coords) + xi_offset;
    const auto& y = get<1>(coords);
    const auto& z = get<2>(coords);
    Variables<tmpl::list<VectorTag<3>>> vars(x.size());
    get<0>(get<VectorTag<3>>(vars)) = 0.4 * x * y * z + square(z);
    get<1>(get<VectorTag<3>>(vars)) = z + 0.1 * square(x);
    get<2>(get<VectorTag<3>>(vars)) = y + square(x) * z;
    return vars;
  };
  const auto make_lower_eta_vars =
      [](const tnsr::I<DataVector, 3, Frame::Logical>& coords,
         const double eta_offset = 0.0) noexcept {
    const auto& x = get<0>(coords);
    const auto y = get<1>(coords) + eta_offset;
    const auto& z = get<2>(coords);
    Variables<tmpl::list<VectorTag<3>>> vars(x.size());
    get<0>(get<VectorTag<3>>(vars)) = 0.4 * x * y * z + square(z);
    get<1>(get<VectorTag<3>>(vars)) = -0.1 * y + z;
    get<2>(get<VectorTag<3>>(vars)) = -square(z);
    return vars;
  };
  const auto make_upper_eta_vars =
      [](const tnsr::I<DataVector, 3, Frame::Logical>& coords,
         const double eta_offset = 0.0) noexcept {
    const auto& x = get<0>(coords);
    const auto y = get<1>(coords) + eta_offset;
    const auto& z = get<2>(coords);
    Variables<tmpl::list<VectorTag<3>>> vars(x.size());
    get<0>(get<VectorTag<3>>(vars)) = 0.4 * x * y * z + square(z);
    get<1>(get<VectorTag<3>>(vars)) = z + 0.4 * x * cube(z);
    get<2>(get<VectorTag<3>>(vars)) = y * z + square(y) + cube(z);
    return vars;
  };
  const auto make_lower_zeta_vars =
      [](const tnsr::I<DataVector, 3, Frame::Logical>& coords,
         const double zeta_offset = 0.0) noexcept {
    const auto& x = get<0>(coords);
    const auto& y = get<1>(coords);
    const auto z = get<2>(coords) + zeta_offset;
    Variables<tmpl::list<VectorTag<3>>> vars(x.size());
    get<0>(get<VectorTag<3>>(vars)) = 0.4 * x * y * z + square(z);
    get<1>(get<VectorTag<3>>(vars)) = 0.9 * z - 2. * x * z;
    get<2>(get<VectorTag<3>>(vars)) = y + cube(z);
    return vars;
  };
  const auto make_upper_zeta_vars =
      [](const tnsr::I<DataVector, 3, Frame::Logical>& coords,
         const double zeta_offset = 0.0) noexcept {
    const auto& x = get<0>(coords);
    const auto& y = get<1>(coords);
    const auto z = get<2>(coords) + zeta_offset;
    Variables<tmpl::list<VectorTag<3>>> vars(x.size());
    get<0>(get<VectorTag<3>>(vars)) = 0.4 * x * y * z + square(z);
    get<1>(get<VectorTag<3>>(vars)) = 1.3 * square(y) * square(z);
    get<2>(get<VectorTag<3>>(vars)) = -x * y * z + square(y);
    return vars;
  };

  const auto local_data = make_center_tensor(logical_coords);
  VariablesMap<3> neighbor_vars{};
  VariablesMap<3> neighbor_modified_vars{};

  const auto shift_vars_to_local_means = [&mesh, &local_data ](
      const Variables<tmpl::list<VectorTag<3>>>& input) noexcept {
    auto result = input;
    auto& v = get<VectorTag<3>>(result);
    get<0>(v) +=
        mean_value(get<0>(local_data), mesh) - mean_value(get<0>(v), mesh);
    get<1>(v) +=
        mean_value(get<1>(local_data), mesh) - mean_value(get<1>(v), mesh);
    get<2>(v) +=
        mean_value(get<2>(local_data), mesh) - mean_value(get<2>(v), mesh);
    return result;
  };

  const auto make_neighbor_vars =
      [
        &logical_coords, &directions_of_external_boundaries, &neighbor_vars,
        &neighbor_modified_vars, &shift_vars_to_local_means
      ](const std::pair<Direction<3>, ElementId<3>>& neighbor,
        const auto make_vars) noexcept {
    if (directions_of_external_boundaries.count(neighbor.first) == 0) {
      const double offset = (neighbor.first.side() == Side::Lower ? -2.0 : 2.0);
      neighbor_vars[neighbor] = make_vars(logical_coords, offset);
      neighbor_modified_vars[neighbor] =
          shift_vars_to_local_means(make_vars(logical_coords));
    }
  };

  make_neighbor_vars(std::make_pair(Direction<3>::lower_xi(), ElementId<3>(1)),
                     make_lower_xi_vars);

  make_neighbor_vars(std::make_pair(Direction<3>::upper_xi(), ElementId<3>(2)),
                     make_upper_xi_vars);

  make_neighbor_vars(std::make_pair(Direction<3>::lower_eta(), ElementId<3>(3)),
                     make_lower_eta_vars);

  make_neighbor_vars(std::make_pair(Direction<3>::upper_eta(), ElementId<3>(4)),
                     make_upper_eta_vars);

  make_neighbor_vars(
      std::make_pair(Direction<3>::lower_zeta(), ElementId<3>(5)),
      make_lower_zeta_vars);

  make_neighbor_vars(
      std::make_pair(Direction<3>::upper_zeta(), ElementId<3>(6)),
      make_upper_zeta_vars);

  const auto neighbor_data =
      make_neighbor_data_from_neighbor_vars(mesh, element, neighbor_vars);

  // The 3D Simple WENO solution has slightly larger numerical error, presumably
  // arising from the 3D extrapolation
  Approx custom_approx = Approx::custom().epsilon(1.e-11).scale(1.0);
  test_simple_weno_work<3>(local_data, mesh, element, neighbor_data,
                           neighbor_modified_vars, custom_approx);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.Limiters.SimpleWenoImpl",
                  "[Limiters][Unit]") {
  test_simple_weno_1d();
  test_simple_weno_2d();
  test_simple_weno_3d();

  // Test with particular boundaries labeled as external
  test_simple_weno_1d({{Direction<1>::lower_xi()}});
  test_simple_weno_2d({{Direction<2>::lower_eta()}});
  test_simple_weno_2d({{Direction<2>::lower_xi(), Direction<2>::lower_eta(),
                        Direction<2>::upper_eta()}});
  test_simple_weno_3d({{Direction<3>::lower_zeta()}});
  test_simple_weno_3d({{Direction<3>::lower_xi(), Direction<3>::upper_xi(),
                        Direction<3>::lower_eta(), Direction<3>::lower_zeta(),
                        Direction<3>::lower_zeta()}});
}
