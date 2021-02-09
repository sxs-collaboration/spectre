// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg {
namespace {
template <size_t Dim>
void test_global_time_stepping_usage() {
  std::uniform_real_distribution<double> dist(-1.0, 2.3);
  MAKE_GENERATOR(gen);
  constexpr size_t number_of_components = 1 + Dim;

  MortarData<Dim> mortar_data{};
  const TimeStepId time_step_id{true, 3, Time{Slab{0.2, 7.1}, {2, 51}}};

  const Mesh<Dim - 1> mortar_mesh{4, Spectral::Basis::Legendre,
                                  Spectral::Quadrature::Gauss};

  const Mesh<Dim - 1> local_mesh{4, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto};
  std::vector<double> local_data(mortar_mesh.number_of_grid_points() *
                                 number_of_components);
  fill_with_random_values(make_not_null(&local_data), make_not_null(&gen),
                          make_not_null(&dist));

  const Mesh<Dim - 1> neighbor_mesh{3, Spectral::Basis::Legendre,
                                    Spectral::Quadrature::Gauss};
  std::vector<double> neighbor_data(mortar_mesh.number_of_grid_points() *
                                    number_of_components);
  fill_with_random_values(make_not_null(&local_data), make_not_null(&gen),
                          make_not_null(&dist));

  CHECK_FALSE(mortar_data.local_mortar_data().has_value());
  CHECK_FALSE(mortar_data.neighbor_mortar_data().has_value());

  mortar_data.insert_local_mortar_data(time_step_id, local_mesh, local_data);

  CHECK(mortar_data.local_mortar_data().has_value());
  CHECK_FALSE(mortar_data.neighbor_mortar_data().has_value());

  mortar_data.insert_neighbor_mortar_data(time_step_id, neighbor_mesh,
                                          neighbor_data);
  CHECK(mortar_data.local_mortar_data().has_value());
  CHECK(mortar_data.neighbor_mortar_data().has_value());

  const auto deserialized_mortar_data = serialize_and_deserialize(mortar_data);

  CHECK(*mortar_data.local_mortar_data() ==
        *deserialized_mortar_data.local_mortar_data());
  CHECK(*mortar_data.neighbor_mortar_data() ==
        *deserialized_mortar_data.neighbor_mortar_data());

  CHECK(mortar_data == deserialized_mortar_data);
  CHECK_FALSE(mortar_data != deserialized_mortar_data);

  const auto extracted_data = mortar_data.extract();
  CHECK(extracted_data.first == *deserialized_mortar_data.local_mortar_data());
  CHECK(extracted_data.second ==
        *deserialized_mortar_data.neighbor_mortar_data());
}

template <size_t Dim>
void test_local_time_stepping_usage(const bool use_gauss_points) {
  CAPTURE(Dim);
  CAPTURE(use_gauss_points);
  std::uniform_real_distribution<double> dist(-1.0, 2.3);
  MAKE_GENERATOR(gen);
  constexpr size_t number_of_components = 1 + Dim;

  MortarData<Dim> mortar_data{};
  const TimeStepId time_step_id{true, 3, Time{Slab{0.2, 7.1}, {2, 51}}};

  const Mesh<Dim - 1> mortar_mesh{4, Spectral::Basis::Legendre,
                                  Spectral::Quadrature::Gauss};

  const Mesh<Dim - 1> local_mesh{4, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto};
  std::vector<double> local_data(mortar_mesh.number_of_grid_points() *
                                 number_of_components);
  fill_with_random_values(make_not_null(&local_data), make_not_null(&gen),
                          make_not_null(&dist));

  CHECK_FALSE(mortar_data.local_mortar_data().has_value());
  CHECK_FALSE(mortar_data.neighbor_mortar_data().has_value());

  mortar_data.insert_local_mortar_data(time_step_id, local_mesh, local_data);

  CHECK(mortar_data.local_mortar_data().has_value());
  CHECK_FALSE(mortar_data.neighbor_mortar_data().has_value());

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

  if (use_gauss_points) {
    mortar_data.insert_local_geometric_quantities(local_volume_det_inv_jacobian,
                                                  local_face_det_jacobian,
                                                  local_face_normal_magnitude);
  } else {
    mortar_data.insert_local_face_normal_magnitude(local_face_normal_magnitude);
  }

  const auto check_geometric_quantities = [&local_face_det_jacobian,
                                           &local_volume_det_inv_jacobian,
                                           &local_face_normal_magnitude,
                                           use_gauss_points](
                                              const auto& mortar_data_local) {
    if (use_gauss_points) {
      Scalar<DataVector> retrieved_local_face_det_jacobian{};
      Scalar<DataVector> retrieved_local_volume_det_inv_jacobian{};
      mortar_data_local.get_local_face_det_jacobian(
          &retrieved_local_face_det_jacobian);
      mortar_data_local.get_local_volume_det_inv_jacobian(
          &retrieved_local_volume_det_inv_jacobian);
      CHECK(retrieved_local_face_det_jacobian == local_face_det_jacobian);
      CHECK(retrieved_local_volume_det_inv_jacobian ==
            local_volume_det_inv_jacobian);
    }
    Scalar<DataVector> retrieved_local_face_normal_magnitude{};
    mortar_data_local.get_local_face_normal_magnitude(
        &retrieved_local_face_normal_magnitude);
    CHECK(retrieved_local_face_normal_magnitude == local_face_normal_magnitude);
  };

  check_geometric_quantities(mortar_data);

  const auto deserialized_mortar_data = serialize_and_deserialize(mortar_data);

  CHECK(*mortar_data.local_mortar_data() ==
        *deserialized_mortar_data.local_mortar_data());
  CHECK_FALSE(deserialized_mortar_data.neighbor_mortar_data().has_value());
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
