// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Elasticity/Equations.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <size_t Dim>
void primal_fluxes(
    // This wrapper function is needed to construct a constitutive relation for
    // the random values of its parameters.
    const gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
        flux_for_displacement,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& strain,
    const tnsr::I<DataVector, Dim>& coordinates, const double bulk_modulus,
    const double shear_modulus) noexcept {
  Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>
      constitutive_relation{bulk_modulus, shear_modulus};
  Elasticity::primal_fluxes<Dim>(flux_for_displacement, strain,
                                 std::move(constitutive_relation), coordinates);
}

template <size_t Dim>
void test_equations(const DataVector& used_for_size) {
  pypp::check_with_random_values<4>(
      &primal_fluxes<Dim>, "Equations",
      {MakeString{} << "primal_fluxes_" << Dim << "d"},
      {{{-1., 1.}, {-1., 1.}, {0., 1.}, {0., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &Elasticity::add_curved_sources<Dim>, "Equations", {"add_curved_sources"},
      {{{-1., 1.}}}, used_for_size, 1.e-12, {}, 0.);
  pypp::check_with_random_values<1>(&Elasticity::auxiliary_fluxes<Dim>,
                                    "Equations", {"auxiliary_fluxes"},
                                    {{{-1., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(&Elasticity::curved_auxiliary_fluxes<Dim>,
                                    "Equations", {"curved_auxiliary_fluxes"},
                                    {{{-1., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &Elasticity::add_curved_auxiliary_sources<Dim>, "Equations",
      {"add_curved_auxiliary_sources"}, {{{-1., 1.}}}, used_for_size, 1.e-12,
      {}, 0.);
}

template <size_t Dim>
void test_computers(const DataVector& used_for_size) {
  using field_tag = Elasticity::Tags::Displacement<Dim>;
  using auxiliary_field_tag = Elasticity::Tags::Strain<Dim>;
  using field_flux_tag =
      ::Tags::Flux<field_tag, tmpl::size_t<Dim>, Frame::Inertial>;
  using auxiliary_flux_tag =
      ::Tags::Flux<auxiliary_field_tag, tmpl::size_t<Dim>, Frame::Inertial>;
  using field_source_tag = ::Tags::Source<field_tag>;
  using auxiliary_source_tag = ::Tags::Source<auxiliary_field_tag>;
  using constitutive_relation_tag = Elasticity::Tags::ConstitutiveRelation<Dim>;
  using coordinates_tag = domain::Tags::Coordinates<Dim, Frame::Inertial>;
  const size_t num_points = used_for_size.size();
  {
    INFO("Fluxes" << Dim << "D");
    std::unique_ptr<
        Elasticity::ConstitutiveRelations::ConstitutiveRelation<Dim>>
        constitutive_relation = std::make_unique<
            Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>>(1.,
                                                                          2.);
    auto box = db::create<db::AddSimpleTags<
        field_tag, auxiliary_field_tag, field_flux_tag, auxiliary_flux_tag,
        constitutive_relation_tag, coordinates_tag>>(
        tnsr::I<DataVector, Dim>{num_points, 1.},
        tnsr::ii<DataVector, Dim>{num_points, 2.},
        tnsr::IJ<DataVector, Dim>{num_points,
                                  std::numeric_limits<double>::signaling_NaN()},
        tnsr::Ijj<DataVector, Dim>{
            num_points, std::numeric_limits<double>::signaling_NaN()},
        std::move(constitutive_relation),
        tnsr::I<DataVector, Dim>{num_points, 6.});

    const Elasticity::Fluxes<Dim> fluxes_computer{};
    using argument_tags = typename Elasticity::Fluxes<Dim>::argument_tags;

    db::mutate_apply<tmpl::list<field_flux_tag>, argument_tags>(
        fluxes_computer, make_not_null(&box), get<auxiliary_field_tag>(box));
    auto expected_field_flux = tnsr::IJ<DataVector, Dim>{
        num_points, std::numeric_limits<double>::signaling_NaN()};
    Elasticity::primal_fluxes(
        make_not_null(&expected_field_flux), get<auxiliary_field_tag>(box),
        get<constitutive_relation_tag>(box), get<coordinates_tag>(box));
    CHECK(get<field_flux_tag>(box) == expected_field_flux);

    db::mutate_apply<tmpl::list<auxiliary_flux_tag>, argument_tags>(
        fluxes_computer, make_not_null(&box), get<field_tag>(box));
    auto expected_auxiliary_flux = tnsr::Ijj<DataVector, Dim>{
        num_points, std::numeric_limits<double>::signaling_NaN()};
    Elasticity::auxiliary_fluxes(make_not_null(&expected_auxiliary_flux),
                                 get<field_tag>(box));
    CHECK(get<auxiliary_flux_tag>(box) == expected_auxiliary_flux);
  }
  {
    INFO("Sources" << Dim << "D");
    auto box =
        db::create<db::AddSimpleTags<field_source_tag, auxiliary_source_tag>>(
            tnsr::I<DataVector, Dim>{num_points, 0.},
            tnsr::ii<DataVector, Dim>{num_points, 0.});

    const Elasticity::Sources<Dim> sources_computer{};
    using argument_tags = typename Elasticity::Sources<Dim>::argument_tags;

    db::mutate_apply<tmpl::list<auxiliary_source_tag>, argument_tags>(
        sources_computer, make_not_null(&box),
        tnsr::I<DataVector, Dim>{num_points,
                                 std::numeric_limits<double>::signaling_NaN()});
    db::mutate_apply<tmpl::list<field_source_tag>, argument_tags>(
        sources_computer, make_not_null(&box),
        tnsr::I<DataVector, Dim>{num_points,
                                 std::numeric_limits<double>::signaling_NaN()},
        tnsr::IJ<DataVector, Dim>{
            num_points, std::numeric_limits<double>::signaling_NaN()});
    auto expected_field_source = tnsr::I<DataVector, Dim>{num_points, 0.};
    auto expected_auxiliary_source = tnsr::ii<DataVector, Dim>{num_points, 0.};
    CHECK(get<field_source_tag>(box) == expected_field_source);
    CHECK(get<auxiliary_source_tag>(box) == expected_auxiliary_source);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.Elasticity", "[Unit][Elliptic]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Elliptic/Systems/Elasticity"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_equations, (2, 3));
  CHECK_FOR_DATAVECTORS(test_computers, (2, 3));
}
