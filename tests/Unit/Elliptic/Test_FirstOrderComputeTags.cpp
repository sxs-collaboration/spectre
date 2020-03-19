// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Elliptic/FirstOrderComputeTags.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/DataStructures/DataBox/TestHelpers.hpp"

namespace {

struct FieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct AuxiliaryFieldTag : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim>;
};

struct AnArgument : db::SimpleTag {
  using type = double;
};

struct BaseArgumentTag : db::BaseTag {};

struct DerivedArgumentTag : BaseArgumentTag, db::SimpleTag {
  using type = double;
};

template <size_t Dim>
struct Fluxes {
  using argument_tags = tmpl::list<AnArgument, BaseArgumentTag>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
      const double an_argument, const double base_tag_argument,
      const tnsr::i<DataVector, Dim>& auxiliary_field) {
    for (size_t d = 0; d < Dim; d++) {
      flux_for_field->get(d) =
          auxiliary_field.get(d) * an_argument + base_tag_argument;
    }
  }
  static void apply(
      const gsl::not_null<tnsr::Ij<DataVector, Dim>*> flux_for_aux_field,
      const double an_argument, const double base_tag_argument,
      const Scalar<DataVector>& field) {
    std::fill(flux_for_aux_field->begin(), flux_for_aux_field->end(), 0.);
    for (size_t d = 0; d < Dim; d++) {
      flux_for_aux_field->get(d, d) =
          get(field) * an_argument + base_tag_argument;
    }
  }
};

struct Sources {
  using argument_tags = tmpl::list<AnArgument, BaseArgumentTag>;
  static void apply(const gsl::not_null<Scalar<DataVector>*> source_for_field,
                    const double an_argument, const double base_tag_argument,
                    const Scalar<DataVector>& field) {
    get(*source_for_field) =
        get(field) * square(an_argument) + base_tag_argument;
  }
};

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using variables_tag =
      Tags::Variables<tmpl::list<FieldTag, AuxiliaryFieldTag<Dim>>>;
  using primal_variables = tmpl::list<FieldTag>;
  using auxiliary_variables = tmpl::list<AuxiliaryFieldTag<Dim>>;
  using fluxes = Fluxes<Dim>;
  using sources = Sources;
};

template <size_t Dim>
void test_first_order_compute_tags() {
  using system = System<Dim>;
  using vars_tag = typename system::variables_tag;
  using FluxesComputer = typename system::fluxes;
  using fluxes_computer_tag = elliptic::Tags::FluxesComputer<FluxesComputer>;
  using first_order_fluxes_compute_tag =
      elliptic::Tags::FirstOrderFluxesCompute<system>;
  using first_order_sources_compute_tag =
      elliptic::Tags::FirstOrderSourcesCompute<system>;

  TestHelpers::db::test_compute_tag<first_order_fluxes_compute_tag>(
      "Flux(Variables(Flux(FieldTag),Flux(AuxiliaryFieldTag)))");
  TestHelpers::db::test_compute_tag<first_order_sources_compute_tag>(
      "Source(Variables(Source(FieldTag),Source(AuxiliaryFieldTag)))");

  // Construct some field data
  static constexpr size_t num_points = 3;
  db::item_type<vars_tag> vars{num_points, 0.};
  get(get<FieldTag>(vars)) = DataVector{num_points, 2.};
  for (size_t d = 0; d < Dim; d++) {
    get<AuxiliaryFieldTag<Dim>>(vars).get(d) = DataVector{num_points, d + 1.};
  }

  // Construct DataBox for testing the compute tags
  const auto box =
      db::create<db::AddSimpleTags<vars_tag, fluxes_computer_tag, AnArgument,
                                   DerivedArgumentTag>,
                 db::AddComputeTags<first_order_fluxes_compute_tag,
                                    first_order_sources_compute_tag>>(
          std::move(vars), FluxesComputer{}, 3., 1.);

  // Check computed fluxes
  for (size_t d = 0; d < Dim; d++) {
    CHECK(get<::Tags::Flux<FieldTag, tmpl::size_t<Dim>, Frame::Inertial>>(box)
              .get(d) == DataVector{num_points, 3. * (d + 1.) + 1.});
    CHECK(get<::Tags::Flux<AuxiliaryFieldTag<Dim>, tmpl::size_t<Dim>,
                           Frame::Inertial>>(box)
              .get(d, d) == DataVector{num_points, 7.});
    for (size_t i0 = 0; i0 < Dim; i0++) {
      if (i0 != d) {
        CHECK(get<::Tags::Flux<AuxiliaryFieldTag<Dim>, tmpl::size_t<Dim>,
                               Frame::Inertial>>(box)
                  .get(d, i0) == DataVector{num_points, 0.});
      }
    }
  }

  // Check computed sources
  CHECK(get(get<::Tags::Source<FieldTag>>(box)) == DataVector{num_points, 19.});
  for (size_t d = 0; d < Dim; d++) {
    CHECK(get<::Tags::Source<AuxiliaryFieldTag<Dim>>>(box).get(d) ==
          get<AuxiliaryFieldTag<Dim>>(box).get(d));
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.FirstOrderComputeTags", "[Unit][Elliptic]") {
  test_first_order_compute_tags<1>();
  test_first_order_compute_tags<2>();
  test_first_order_compute_tags<3>();
}
