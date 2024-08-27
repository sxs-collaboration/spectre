// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg {
namespace {
template <size_t Dim>
void assign_with_reference(const gsl::not_null<MortarData<Dim>*> mortar_data,
                           const Mesh<Dim - 1>& mortar_mesh,
                           const Mesh<Dim - 1>& face_mesh,
                           const std::optional<DataVector>& data,
                           const std::string& expected_output) {
  if (data.has_value()) {
    mortar_data->mortar_mesh = mortar_mesh;
    mortar_data->face_mesh = face_mesh;
    mortar_data->mortar_data = *data;
    CHECK(mortar_data->mortar_data.has_value());
  }

  CHECK(get_output(*mortar_data) == expected_output);
}

template <size_t Dim>
void check_serialization(const gsl::not_null<MortarData<Dim>*> mortar_data) {
  const auto deserialized_mortar_data = serialize_and_deserialize(*mortar_data);

  CHECK(*mortar_data->mortar_data == *deserialized_mortar_data.mortar_data);

  CHECK(*mortar_data == deserialized_mortar_data);
  CHECK_FALSE(*mortar_data != deserialized_mortar_data);

  CHECK(*mortar_data->mortar_data == *deserialized_mortar_data.mortar_data);
}

template <size_t Dim>
void test_global_time_stepping_usage() {
  std::uniform_real_distribution<double> dist(-1.0, 2.3);
  MAKE_GENERATOR(gen);
  constexpr size_t number_of_components = 1 + Dim;

  MortarData<Dim> local_mortar_data{};

  const Mesh<Dim - 1> mortar_mesh{4, Spectral::Basis::Legendre,
                                  Spectral::Quadrature::Gauss};

  const Mesh<Dim - 1> local_mesh{4, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto};
  DataVector local_data{
      mortar_mesh.number_of_grid_points() * number_of_components, 0.0};
  fill_with_random_values(make_not_null(&local_data), make_not_null(&gen),
                          make_not_null(&dist));

  MortarData<Dim> neighbor_mortar_data{};

  const Mesh<Dim - 1> neighbor_mesh{3, Spectral::Basis::Legendre,
                                    Spectral::Quadrature::Gauss};
  DataVector neighbor_data{
      mortar_mesh.number_of_grid_points() * number_of_components, 0.0};
  fill_with_random_values(make_not_null(&neighbor_data), make_not_null(&gen),
                          make_not_null(&dist));

  std::string local_expected_output = MakeString{}
                                      << "Mortar data: " << local_data << "\n"
                                      << "Mortar mesh: " << mortar_mesh << "\n"
                                      << "Face normal magnitude: --\n"
                                      << "Face det(J): --\n"
                                      << "Face mesh: " << local_mesh << "\n"
                                      << "Volume det(invJ): --\n"
                                      << "Volume mesh: --\n";

  std::string neighbor_expected_output =
      MakeString{} << "Mortar data: " << neighbor_data << "\n"
                   << "Mortar mesh: " << mortar_mesh << "\n"
                   << "Face normal magnitude: --\n"
                   << "Face det(J): --\n"
                   << "Face mesh: " << neighbor_mesh << "\n"
                   << "Volume det(invJ): --\n"
                   << "Volume mesh: --\n";

  CHECK_FALSE(local_mortar_data.mortar_data.has_value());
  CHECK_FALSE(neighbor_mortar_data.mortar_data.has_value());

  assign_with_reference(make_not_null(&local_mortar_data), mortar_mesh,
                        local_mesh, std::optional{local_data},
                        local_expected_output);

  assign_with_reference(make_not_null(&neighbor_mortar_data), mortar_mesh,
                        neighbor_mesh, std::optional{neighbor_data},
                        neighbor_expected_output);

  check_serialization(make_not_null(&local_mortar_data));
  check_serialization(make_not_null(&neighbor_mortar_data));
}

template <size_t Dim>
void test_local_time_stepping_usage(const bool use_gauss_points) {
  CAPTURE(Dim);
  CAPTURE(use_gauss_points);
  std::uniform_real_distribution<double> dist(-1.0, 2.3);
  MAKE_GENERATOR(gen);
  constexpr size_t number_of_components = 1 + Dim;

  MortarData<Dim> mortar_data{};
  const Mesh<Dim - 1> mortar_mesh{4, Spectral::Basis::Legendre,
                                  Spectral::Quadrature::Gauss};

  const Mesh<Dim - 1> local_mesh{4, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto};
  DataVector local_data{
      mortar_mesh.number_of_grid_points() * number_of_components, 0.0};
  fill_with_random_values(make_not_null(&local_data), make_not_null(&gen),
                          make_not_null(&dist));

  std::string expected_output = MakeString{}
                                << "Mortar data: " << local_data << "\n"
                                << "Mortar mesh: " << mortar_mesh << "\n"
                                << "Face normal magnitude: --\n"
                                << "Face det(J): --\n"
                                << "Face mesh: " << local_mesh << "\n"
                                << "Volume det(invJ): --\n"
                                << "Volume mesh: --\n";

  assign_with_reference(make_not_null(&mortar_data), mortar_mesh, local_mesh,
                        std::optional{local_data}, expected_output);

  const auto local_volume_det_inv_jacobian =
      make_with_random_values<Scalar<DataVector>>(
          make_not_null(&gen), make_not_null(&dist),
          mortar_mesh.number_of_grid_points() * 4);
  const auto local_face_det_jacobian =
      make_with_random_values<Scalar<DataVector>>(
          make_not_null(&gen), make_not_null(&dist),
          mortar_mesh.number_of_grid_points());
  const auto local_face_normal_magnitude =
      make_with_random_values<Scalar<DataVector>>(
          make_not_null(&gen), make_not_null(&dist),
          mortar_mesh.number_of_grid_points());

  mortar_data.face_normal_magnitude = local_face_normal_magnitude;
  if (use_gauss_points) {
    mortar_data.volume_det_inv_jacobian = local_volume_det_inv_jacobian;
    mortar_data.face_det_jacobian = local_face_det_jacobian;
  }

  const auto check_geometric_quantities = [&local_face_det_jacobian,
                                           &local_volume_det_inv_jacobian,
                                           &local_face_normal_magnitude,
                                           use_gauss_points](
                                              const auto& mortar_data_local) {
    if (use_gauss_points) {
      CHECK(mortar_data_local.face_det_jacobian == local_face_det_jacobian);
      CHECK(mortar_data_local.volume_det_inv_jacobian ==
            local_volume_det_inv_jacobian);
    }
    CHECK(mortar_data_local.face_normal_magnitude ==
          local_face_normal_magnitude);
  };

  check_geometric_quantities(mortar_data);

  // We don't use the check_serialization function from above because we didn't
  // insert neighbor data here
  const auto deserialized_mortar_data = serialize_and_deserialize(mortar_data);

  CHECK(*mortar_data.mortar_data == *deserialized_mortar_data.mortar_data);
  check_geometric_quantities(deserialized_mortar_data);

  CHECK(mortar_data == deserialized_mortar_data);
  CHECK_FALSE(mortar_data != deserialized_mortar_data);
}

template <size_t Dim>
std::array<size_t, Dim> make_extents(const size_t first_extent) {
  if constexpr (1 == Dim) {
    return {first_extent};
  } else if constexpr (2 == Dim) {
    return {first_extent, first_extent + 1};
  } else if constexpr (3 == Dim) {
    return {first_extent, first_extent + 1, first_extent + 2};
  } else {
    return {};
  }
}

template <size_t Dim>
void check_mortar_data(const MortarData<Dim>& projected,
                       const MortarData<Dim>& expected) {
  CHECK(projected.mortar_mesh == expected.mortar_mesh);
  CHECK(projected.face_mesh == expected.face_mesh);
  CHECK(projected.volume_mesh == expected.volume_mesh);
  if (projected.mortar_data.has_value()) {
    CHECK_ITERABLE_APPROX(projected.mortar_data.value(),
                          expected.mortar_data.value());
  } else {
    CHECK_FALSE(expected.mortar_data.has_value());
  }
  if (projected.face_normal_magnitude.has_value()) {
    CHECK_ITERABLE_APPROX(projected.face_normal_magnitude.value(),
                          expected.face_normal_magnitude.value());
  } else {
    CHECK_FALSE(expected.face_normal_magnitude.has_value());
  }
  if (projected.face_det_jacobian.has_value()) {
    CHECK_ITERABLE_APPROX(projected.face_det_jacobian.value(),
                          expected.face_det_jacobian.value());
  } else {
    CHECK_FALSE(expected.face_det_jacobian.has_value());
  }
  if (projected.volume_det_inv_jacobian.has_value()) {
    CHECK_ITERABLE_APPROX(projected.volume_det_inv_jacobian.value(),
                          expected.volume_det_inv_jacobian.value());
  } else {
    CHECK_FALSE(expected.volume_det_inv_jacobian.has_value());
  }
}

template <size_t Dim>
void test_p_project() {
  const Mesh<Dim> initial_volume_mesh{make_extents<Dim>(3),
                                      Spectral::Basis::Legendre,
                                      Spectral::Quadrature::Gauss};
  const Mesh<Dim - 1> initial_face_mesh =
      initial_volume_mesh.slice_away(Dim - 1);
  const Mesh<Dim - 1> initial_mortar_mesh = initial_volume_mesh.slice_away(0);
  const Mesh<Dim> final_volume_mesh{make_extents<Dim>(4),
                                    Spectral::Basis::Legendre,
                                    Spectral::Quadrature::Gauss};
  const Mesh<Dim - 1> final_face_mesh = final_volume_mesh.slice_away(Dim - 1);
  const Mesh<Dim - 1> final_mortar_mesh = final_volume_mesh.slice_away(0);
  constexpr size_t number_of_components = 1 + Dim;
  const DataVector initial_mortar_data{
      initial_mortar_mesh.number_of_grid_points() * number_of_components, 1.0};
  const Scalar<DataVector> initial_face_normal_magnitude{
      DataVector{initial_face_mesh.number_of_grid_points(), 2.0}};
  const Scalar<DataVector> initial_face_det_jacobian{
      DataVector{initial_face_mesh.number_of_grid_points(), 3.0}};
  const Scalar<DataVector> initial_volume_det_inv_jacobian{
      DataVector{initial_volume_mesh.number_of_grid_points(), 4.0}};
  const DataVector final_mortar_data{
      final_mortar_mesh.number_of_grid_points() * number_of_components, 1.0};
  const Scalar<DataVector> final_face_normal_magnitude{
      DataVector{final_face_mesh.number_of_grid_points(), 2.0}};
  const Scalar<DataVector> final_face_det_jacobian{
      DataVector{final_face_mesh.number_of_grid_points(), 3.0}};
  const Scalar<DataVector> final_volume_det_inv_jacobian{
      DataVector{final_volume_mesh.number_of_grid_points(), 4.0}};
  MortarData<Dim> only_mortar_data{std::optional(initial_mortar_data),
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::optional(initial_mortar_mesh),
                                   std::nullopt,
                                   std::nullopt};
  p_project(make_not_null(&only_mortar_data), final_mortar_mesh,
            final_face_mesh, final_volume_mesh);
  check_mortar_data(only_mortar_data,
                    MortarData<Dim>{std::optional(final_mortar_data),
                                    std::nullopt, std::nullopt, std::nullopt,
                                    std::optional(final_mortar_mesh),
                                    std::nullopt, std::nullopt});
  MortarData<Dim> only_mortar_data_2{std::optional(initial_mortar_data),
                                     std::nullopt,
                                     std::nullopt,
                                     std::nullopt,
                                     std::optional(initial_mortar_mesh),
                                     std::nullopt,
                                     std::nullopt};
  p_project_only_mortar_data(make_not_null(&only_mortar_data_2),
                             final_mortar_mesh);
  check_mortar_data(only_mortar_data_2,
                    MortarData<Dim>{std::optional(final_mortar_data),
                                    std::nullopt, std::nullopt, std::nullopt,
                                    std::optional(final_mortar_mesh),
                                    std::nullopt, std::nullopt});
  MortarData<Dim> only_gl_data{std::optional(initial_mortar_data),
                               std::optional(initial_face_normal_magnitude),
                               std::nullopt,
                               std::nullopt,
                               std::optional(initial_mortar_mesh),
                               std::optional(initial_face_mesh),
                               std::nullopt};
  p_project(make_not_null(&only_gl_data), final_mortar_mesh, final_face_mesh,
            final_volume_mesh);
  check_mortar_data(
      only_gl_data,
      MortarData<Dim>{std::optional(final_mortar_data),
                      std::optional(final_face_normal_magnitude), std::nullopt,
                      std::nullopt, std::optional(final_mortar_mesh),
                      std::optional(final_face_mesh), std::nullopt});
  MortarData<Dim> g_data{std::optional(initial_mortar_data),
                         std::optional(initial_face_normal_magnitude),
                         std::optional(initial_face_det_jacobian),
                         std::optional(initial_volume_det_inv_jacobian),
                         std::optional(initial_mortar_mesh),
                         std::optional(initial_face_mesh),
                         std::optional(initial_volume_mesh)};
  p_project(make_not_null(&g_data), final_mortar_mesh, final_face_mesh,
            final_volume_mesh);
  check_mortar_data(
      g_data, MortarData<Dim>{std::optional(final_mortar_data),
                              std::optional(final_face_normal_magnitude),
                              std::optional(final_face_det_jacobian),
                              std::optional(final_volume_det_inv_jacobian),
                              std::optional(final_mortar_mesh),
                              std::optional(final_face_mesh),
                              std::optional(final_volume_mesh)});
}

template <size_t Dim>
void test() {
  test_global_time_stepping_usage<Dim>();
  test_local_time_stepping_usage<Dim>(true);
  test_local_time_stepping_usage<Dim>(false);
  test_p_project<Dim>();
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.MortarData", "[Unit][Evolution]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace evolution::dg
