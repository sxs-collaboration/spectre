// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Domain/Tags/Faces.hpp"
#include "Elliptic/BoundaryConditions/Tags/BoundaryFields.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

SPECTRE_TEST_CASE("Unit.Elliptic.BoundaryConditions.BoundaryFields",
                  "[Unit][Elliptic]") {
  static constexpr size_t Dim = 2;
  using vars_tag = ::Tags::Variables<tmpl::list<::Tags::TempScalar<0>>>;
  using fluxes_tag = ::Tags::Variables<tmpl::list<::Tags::TempI<0, Dim>>>;
  TestHelpers::db::test_compute_tag<
      elliptic::Tags::BoundaryFieldsCompute<Dim, vars_tag>>(
      "Faces(Variables(TempTensor0))");
  TestHelpers::db::test_compute_tag<
      elliptic::Tags::BoundaryFluxesCompute<Dim, vars_tag, fluxes_tag>>(
      "Faces(Variables(NormalDotFlux(TempTensor0)))");
  {
    const Mesh<Dim> mesh{3, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};
    const Element<Dim> element{
        ElementId<Dim>{0},
        {{Direction<Dim>::upper_xi(), {{{ElementId<Dim>{1}}}, {}}},
         {Direction<Dim>::lower_eta(), {{{ElementId<Dim>{1}}}, {}}},
         {Direction<Dim>::upper_eta(), {{{ElementId<Dim>{1}}}, {}}}}};
    tnsr::i<DataVector, Dim> face_normal{size_t{3}, 0.};
    get<0>(face_normal) = -1.;
    DirectionMap<Dim, tnsr::i<DataVector, Dim>> face_normals{
        {Direction<Dim>::lower_xi(), std::move(face_normal)}};
    Variables<tmpl::list<::Tags::TempScalar<0>>> vars{size_t{9}, 0.};
    auto& var = get(get<::Tags::TempScalar<0>>(vars));
    std::iota(var.begin(), var.end(), 1.);
    Variables<tmpl::list<::Tags::TempI<0, Dim>>> fluxes{size_t{9}, 0.};
    for (size_t d = 0; d < Dim; ++d) {
      auto& flux = get<::Tags::TempI<0, Dim>>(fluxes).get(d);
      std::iota(flux.begin(), flux.end(), 10. * static_cast<double>(d + 1));
    }
    const auto box = db::create<
        db::AddSimpleTags<
            vars_tag, fluxes_tag, domain::Tags::Mesh<Dim>,
            domain::Tags::Element<Dim>,
            domain::Tags::Faces<Dim, domain::Tags::FaceNormal<Dim>>>,
        db::AddComputeTags<
            elliptic::Tags::BoundaryFieldsCompute<Dim, vars_tag>,
            elliptic::Tags::BoundaryFluxesCompute<Dim, vars_tag, fluxes_tag>>>(
        std::move(vars), std::move(fluxes), mesh, element,
        std::move(face_normals));
    CHECK(get(get<domain::Tags::Faces<Dim, ::Tags::TempScalar<0>>>(box).at(
              Direction<Dim>::lower_xi())) == DataVector{1., 4., 7.});
    CHECK(get(get<domain::Tags::Faces<
                  Dim, ::Tags::NormalDotFlux<::Tags::TempScalar<0>>>>(box)
                  .at(Direction<Dim>::lower_xi())) ==
          DataVector{-10., -13., -16.});
  }
}
