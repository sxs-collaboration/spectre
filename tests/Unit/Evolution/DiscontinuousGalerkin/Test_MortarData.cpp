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
                           const Mesh<Dim - 1>& face_mesh,
                           const std::optional<DataVector>& data,
                           const std::string& expected_output) {
  if (data.has_value()) {
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
                                      << "Face normal magnitude: --\n"
                                      << "Face det(J): --\n"
                                      << "Face mesh: " << local_mesh << "\n"
                                      << "Volume det(invJ): --\n";

  std::string neighbor_expected_output =
      MakeString{} << "Mortar data: " << neighbor_data << "\n"
                   << "Face normal magnitude: --\n"
                   << "Face det(J): --\n"
                   << "Face mesh: " << neighbor_mesh << "\n"
                   << "Volume det(invJ): --\n";

  CHECK_FALSE(local_mortar_data.mortar_data.has_value());
  CHECK_FALSE(neighbor_mortar_data.mortar_data.has_value());

  assign_with_reference(make_not_null(&local_mortar_data),
                        local_mesh, std::optional{local_data},
                        local_expected_output);

  assign_with_reference(make_not_null(&neighbor_mortar_data),
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
                                << "Face normal magnitude: --\n"
                                << "Face det(J): --\n"
                                << "Face mesh: " << local_mesh << "\n"
                                << "Volume det(invJ): --\n";

  assign_with_reference(make_not_null(&mortar_data), local_mesh,
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
void test() {
  test_global_time_stepping_usage<Dim>();
  test_local_time_stepping_usage<Dim>(true);
  test_local_time_stepping_usage<Dim>(false);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.MortarData", "[Unit][Evolution]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace evolution::dg
