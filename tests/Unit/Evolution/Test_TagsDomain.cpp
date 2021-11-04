// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/TagsDomain.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim>
void test_tags() {
  TestHelpers::db::test_compute_tag<
      evolution::domain::Tags::DivMeshVelocityCompute<Dim>>(
      "div(MeshVelocity)");
}

template <size_t MeshDim>
using TranslationMap =
    domain::CoordinateMaps::TimeDependent::Translation<MeshDim>;

template <size_t MeshDim>
using ConcreteMap = domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                                          TranslationMap<MeshDim>>;

template <size_t MeshDim>
ConcreteMap<MeshDim> create_coord_map(const std::string& f_of_t_name) {
  return ConcreteMap<MeshDim>{TranslationMap<MeshDim>{f_of_t_name}};
}

template <size_t Dim, bool IsTimeDependent>
void test() {
  using simple_tags = db::AddSimpleTags<
      Tags::Time, domain::Tags::Mesh<Dim>,
      domain::Tags::Coordinates<Dim, Frame::Grid>,
      domain::Tags::InverseJacobian<Dim, Frame::ElementLogical, Frame::Grid>,
      domain::Tags::FunctionsOfTimeInitialize,
      domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                  Frame::Inertial>>;
  using compute_tags = db::AddComputeTags<
      domain::Tags::CoordinatesMeshVelocityAndJacobiansCompute<
          domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                      Frame::Inertial>>,
      domain::Tags::InertialFromGridCoordinatesCompute<Dim>,
      domain::Tags::ElementToInertialInverseJacobian<Dim>,
      domain::Tags::InertialMeshVelocityCompute<Dim>,
      evolution::domain::Tags::DivMeshVelocityCompute<Dim>>;

  const DataVector velocity{Dim, -4.3};
  const double initial_time = 0.0;
  const double expiration_time = 4.5;
  // In 1d, the helper function create_coord_map will only use the first name,
  // i.e., TranslationX. In 2d, it uses the first two names, TranslationX and
  // TranslationY.
  const std::string function_of_time_name = "Translation";

  MAKE_GENERATOR(gen);
  UniformCustomDistribution<double> dist(-10.0, 10.0);

  const Mesh<Dim> mesh{5, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const size_t num_pts = mesh.number_of_grid_points();

  tnsr::I<DataVector, Dim, Frame::Grid> grid_coords{num_pts};
  fill_with_random_values(make_not_null(&grid_coords), make_not_null(&gen),
                          make_not_null(&dist));
  InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Grid>
      element_to_grid_inverse_jacobian{num_pts};
  fill_with_random_values(make_not_null(&element_to_grid_inverse_jacobian),
                          make_not_null(&gen), make_not_null(&dist));

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time[function_of_time_name] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time,
          std::array<DataVector, 3>{{{Dim, 0.0}, velocity, {Dim, 0.0}}},
          expiration_time);

  using MapPtr = std::unique_ptr<
      domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, Dim>>;
  const MapPtr grid_to_inertial_map =
      IsTimeDependent ? MapPtr(std::make_unique<ConcreteMap<Dim>>(
                            create_coord_map<Dim>(function_of_time_name)))
                      : MapPtr(std::make_unique<domain::CoordinateMap<
                                   Frame::Grid, Frame::Inertial,
                                   domain::CoordinateMaps::Identity<Dim>>>());

  const double time = 3.0;
  auto box = db::create<simple_tags, compute_tags>(
      time, mesh, grid_coords, element_to_grid_inverse_jacobian,
      std::move(functions_of_time), grid_to_inertial_map->get_clone());

  const auto check_helper = [&box, &mesh]() {
    if (IsTimeDependent) {
      const std::optional<Scalar<DataVector>>& div_frame_velocity =
          db::get<domain::Tags::DivMeshVelocity>(box);
      REQUIRE(div_frame_velocity.has_value());
      CHECK(
          *div_frame_velocity ==
          divergence(
              db::get<domain::Tags::MeshVelocity<Dim>>(box).value(), mesh,
              db::get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                                    Frame::Inertial>>(box)));
    } else {
      // In the time-independent case, check that the divergence of the mesh
      // velocity is not set.
      CHECK_FALSE(db::get<domain::Tags::DivMeshVelocity>(box).has_value());
    }
  };
  check_helper();

  db::mutate<Tags::Time>(
      make_not_null(&box),
      [](const gsl::not_null<double*> local_time) { *local_time = 4.5; });
  check_helper();
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.TagsDomain", "[Unit][Evolution]") {
  test_tags<1>();
  test_tags<2>();
  test_tags<3>();

  test<1, true>();
  test<2, true>();
  test<3, true>();

  test<1, false>();
  test<2, false>();
  test<3, false>();
}
