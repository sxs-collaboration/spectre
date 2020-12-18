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
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.tpp"
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
void test_tags() noexcept {
  TestHelpers::db::test_compute_tag<
      evolution::domain::Tags::DivMeshVelocityCompute<Dim>>(
      "div(MeshVelocity)");
}

using TranslationMap = domain::CoordinateMaps::TimeDependent::Translation;
using TranslationMap2d =
    domain::CoordinateMaps::TimeDependent::ProductOf2Maps<TranslationMap,
                                                          TranslationMap>;
using TranslationMap3d = domain::CoordinateMaps::TimeDependent::ProductOf3Maps<
    TranslationMap, TranslationMap, TranslationMap>;

using AffineMap = domain::CoordinateMaps::Affine;
using AffineMap2d =
    domain::CoordinateMaps::ProductOf2Maps<AffineMap, AffineMap>;
using AffineMap3d =
    domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;

template <size_t MeshDim>
using ConcreteMap = tmpl::conditional_t<
    MeshDim == 1,
    domain::CoordinateMap<Frame::Grid, Frame::Inertial, TranslationMap,
                          AffineMap>,
    tmpl::conditional_t<MeshDim == 2,
                        domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                                              TranslationMap2d, AffineMap2d>,
                        domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                                              TranslationMap3d, AffineMap3d>>>;

template <size_t MeshDim>
ConcreteMap<MeshDim> create_coord_map(
    const std::array<std::string, 3>& f_of_t_names);

template <>
ConcreteMap<1> create_coord_map<1>(
    const std::array<std::string, 3>& f_of_t_names) {
  return ConcreteMap<1>{TranslationMap{f_of_t_names[0]},
                        AffineMap{-1.0, 1.0, 2.0, 7.2}};
}

template <>
ConcreteMap<2> create_coord_map<2>(
    const std::array<std::string, 3>& f_of_t_names) {
  return ConcreteMap<2>{
      {TranslationMap{f_of_t_names[0]}, TranslationMap{f_of_t_names[1]}},
      {AffineMap{-1.0, 1.0, -2.0, 2.2}, AffineMap{-1.0, 1.0, 2.0, 7.2}}};
}

template <>
ConcreteMap<3> create_coord_map<3>(
    const std::array<std::string, 3>& f_of_t_names) {
  return ConcreteMap<3>{
      {TranslationMap{f_of_t_names[0]}, TranslationMap{f_of_t_names[1]},
       TranslationMap{f_of_t_names[2]}},
      {AffineMap{-1.0, 1.0, -2.0, 2.2}, AffineMap{-1.0, 1.0, 2.0, 7.2},
       AffineMap{-1.0, 1.0, 1.0, 3.5}}};
}

template <size_t Dim, bool IsTimeDependent>
void test() noexcept {
  using simple_tags = db::AddSimpleTags<
      Tags::Time, domain::Tags::Mesh<Dim>,
      domain::Tags::Coordinates<Dim, Frame::Grid>,
      domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Grid>,
      domain::Tags::FunctionsOfTime,
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

  const std::array<double, 3> velocity{{1.2, 0.2, -8.9}};
  const double initial_time = 0.0;
  const double expiration_time = 4.5;
  // In 1d, the helper function create_coord_map will only use the first name,
  // i.e., TranslationX. In 2d, it uses the first two names, TranslationX and
  // TranslationY.
  const std::array<std::string, 3> functions_of_time_names{
      {"TranslationX", "TranslationY", "TranslationZ"}};

  MAKE_GENERATOR(gen);
  UniformCustomDistribution<double> dist(-10.0, 10.0);

  const Mesh<Dim> mesh{5, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const size_t num_pts = mesh.number_of_grid_points();

  tnsr::I<DataVector, Dim, Frame::Grid> grid_coords{num_pts};
  fill_with_random_values(make_not_null(&grid_coords), make_not_null(&gen),
                          make_not_null(&dist));
  InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Grid>
      element_to_grid_inverse_jacobian{num_pts};
  fill_with_random_values(make_not_null(&element_to_grid_inverse_jacobian),
                          make_not_null(&gen), make_not_null(&dist));

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time[functions_of_time_names[0]] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time,
          std::array<DataVector, 3>{{{0.0}, {velocity[0]}, {0.0}}},
          expiration_time);
  functions_of_time[functions_of_time_names[1]] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time,
          std::array<DataVector, 3>{{{0.0}, {velocity[1]}, {0.0}}},
          expiration_time);
  functions_of_time[functions_of_time_names[2]] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time,
          std::array<DataVector, 3>{{{0.0}, {velocity[2]}, {0.0}}},
          expiration_time);

  using MapPtr = std::unique_ptr<
      domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, Dim>>;
  const MapPtr grid_to_inertial_map =
      IsTimeDependent ? MapPtr(std::make_unique<ConcreteMap<Dim>>(
                            create_coord_map<Dim>(functions_of_time_names)))
                      : MapPtr(std::make_unique<domain::CoordinateMap<
                                   Frame::Grid, Frame::Inertial,
                                   domain::CoordinateMaps::Identity<Dim>>>());

  const double time = 3.0;
  auto box = db::create<simple_tags, compute_tags>(
      time, mesh, grid_coords, element_to_grid_inverse_jacobian,
      std::move(functions_of_time), grid_to_inertial_map->get_clone());

  const auto check_helper = [&box, &mesh]() noexcept {
    if (IsTimeDependent) {
      const std::optional<Scalar<DataVector>>& div_frame_velocity =
          db::get<domain::Tags::DivMeshVelocity>(box);
      REQUIRE(static_cast<bool>(div_frame_velocity));
      CHECK(*div_frame_velocity ==
            divergence(
                db::get<domain::Tags::MeshVelocity<Dim>>(box).value(), mesh,
                db::get<domain::Tags::InverseJacobian<Dim, Frame::Logical,
                                                      Frame::Inertial>>(box)));
    } else {
      // In the time-independent case, check that the divergence of the mesh
      // velocity is not set.
      CHECK_FALSE(
          static_cast<bool>(db::get<domain::Tags::DivMeshVelocity>(box)));
    }
  };
  check_helper();

  db::mutate<Tags::Time>(make_not_null(&box),
                         [](const gsl::not_null<double*> local_time) noexcept {
                           *local_time = 4.5;
                         });
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
