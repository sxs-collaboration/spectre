// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/Creators/TimeDependence/RotationAboutZAxis.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/ExcisionSphere.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/InitializeElementFacesGridCoordinates.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace CurvedScalarWave::Worldtube {
namespace {
void test_initialize_element_faces_coordinates_map(
    const DomainCreator<3>& domain_creator,
    const Spectral::Quadrature& quadrature) {
  static constexpr size_t Dim = 3;
  const auto domain = domain_creator.create_domain();
  const auto& excision_spheres = domain.excision_spheres();
  const auto& initial_refinements = domain_creator.initial_refinement_levels();
  const auto& initial_extents = domain_creator.initial_extents();
  for (const auto& [_, excision_sphere] : excision_spheres) {
    auto box = db::create<db::AddSimpleTags<
        ::domain::Tags::Domain<Dim>, ::domain::Tags::InitialExtents<Dim>,
        ::domain::Tags::InitialRefinementLevels<Dim>,
        evolution::dg::Tags::Quadrature, Tags::ExcisionSphere<Dim>,
        Tags::ElementFacesGridCoordinates<Dim>>>(
        domain_creator.create_domain(), initial_extents, initial_refinements,
        quadrature, excision_sphere,
        std::unordered_map<ElementId<3>,
                           tnsr::I<DataVector, Dim, Frame::Grid>>{});

    db::mutate_apply<
        Initialization::InitializeElementFacesGridCoordinates<Dim>>(
        make_not_null(&box));

    const auto& element_faces_coords =
        db::get<Tags::ElementFacesGridCoordinates<Dim>>(box);

    // refinement is the same everywhere so we just get the 0th block in 0th
    // direction
    CHECK(element_faces_coords.size() ==
          6 * pow(4, initial_refinements.at(0).at(0)));
    for (const auto& [element_id, coords] : element_faces_coords) {
      CHECK(excision_sphere.abutting_direction(element_id).has_value());
      const auto& face_size = coords.get(0).size();
      // extents are also the same everywhere
      CHECK(face_size == square(initial_extents.at(0).at(0)));
      CHECK_ITERABLE_APPROX(magnitude(coords).get(),
                            DataVector(face_size, excision_sphere.radius()));
    }
  }
}
}  // namespace

// [[TimeOut, 10]]
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Worldtube.InitializeElementFacesGridCoordinates",
    "[Unit][Evolution]") {
  for (const auto initial_refinement : std::array<size_t, 2>{{0, 1}}) {
    const size_t initial_extents = 3;
    const auto quadrature = Spectral::Quadrature::GaussLobatto;
    CAPTURE(initial_refinement);
    CAPTURE(quadrature);
    INFO("Testing Shell time independent");
    const domain::creators::Sphere shell{1.5,
                                         3.,
                                         domain::creators::Sphere::Excision{},
                                         initial_refinement,
                                         initial_extents,
                                         true};
    test_initialize_element_faces_coordinates_map(shell, quadrature);

    INFO("Testing Shell time dependent");
    const domain::creators::Sphere shell_time_dependent{
        1.5,
        3.,
        domain::creators::Sphere::Excision{},
        initial_refinement,
        initial_extents,
        true,
        std::nullopt,
        {},
        {domain::CoordinateMaps::Distribution::Linear},
        ShellWedges::All,
        std::make_unique<
            domain::creators::time_dependence::RotationAboutZAxis<3>>(0., 1.,
                                                                      1., 1.)};
    test_initialize_element_faces_coordinates_map(shell_time_dependent,
                                                  quadrature);

    INFO("Testing BinaryCompactObject");
    const domain::creators::BinaryCompactObject binary_compact_object{
        domain::creators::BinaryCompactObject::Object{
            0.5, 3., 8.,
            std::make_optional(
                domain::creators::BinaryCompactObject::Excision{nullptr}),
            false},
        domain::creators::BinaryCompactObject::Object{
            1.5, 3., -5.,
            std::make_optional(
                domain::creators::BinaryCompactObject::Excision{nullptr}),
            false},
        30.,
        50.,
        initial_refinement,
        initial_extents};
    test_initialize_element_faces_coordinates_map(binary_compact_object,
                                                  quadrature);
  }
}
}  // namespace CurvedScalarWave::Worldtube
