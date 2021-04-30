// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Creators/Cylinder.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Elasticity/Actions/InitializeConstitutiveRelation.hpp"
#include "Framework/ActionTesting.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Factory.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace Elasticity {
namespace {
constexpr size_t Dim = 3;

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;
  using const_global_cache_tags = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 tmpl::list<domain::Tags::Element<Dim>>>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<Actions::InitializeConstitutiveRelation<Dim>>>>;
};

struct Metavariables {
  using component_list = tmpl::list<ElementArray<Metavariables>>;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<Dim>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        ConstitutiveRelations::ConstitutiveRelation<Dim>,
        ConstitutiveRelations::standard_constitutive_relations<Dim>>>;
  };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Elasticity.Actions.InitializeConstitutiveRelation",
                  "[Unit][Elliptic][Actions]") {
  domain::creators::register_derived_with_charm();
  register_factory_classes_with_charm<Metavariables>();
  // A cylinder with two layers in z-direction
  const std::unique_ptr<DomainCreator<Dim>> domain_creator =
      std::make_unique<domain::creators::Cylinder>(
          1., 3., 0., 10., false, 1_st, 3_st, false, std::vector<double>{},
          std::vector<double>{2.},
          std::vector<domain::CoordinateMaps::Distribution>{
              domain::CoordinateMaps::Distribution::Linear},
          std::vector<domain::CoordinateMaps::Distribution>{
              domain::CoordinateMaps::Distribution::Linear,
              domain::CoordinateMaps::Distribution::Linear});
  const ElementId<Dim> element_id_layer1{0, {{{1, 0}, {1, 0}, {1, 0}}}};
  const ElementId<Dim> element_id_layer2{5, {{{1, 0}, {1, 0}, {1, 0}}}};
  // A different material in each layer
  using ConstRelPtr =
      std::unique_ptr<ConstitutiveRelations::ConstitutiveRelation<Dim>>;
  std::unordered_map<std::string, ConstRelPtr> material_layers{};
  material_layers["Layer0"] =
      std::make_unique<ConstitutiveRelations::IsotropicHomogeneous<Dim>>(1.,
                                                                         2.);
  material_layers["Layer1"] =
      std::make_unique<ConstitutiveRelations::IsotropicHomogeneous<Dim>>(3.,
                                                                         4.);
  const typename OptionTags::ConstitutiveRelationPerBlock<Dim>::type
      material_layers_variant(std::move(material_layers));

  auto material_per_block =
      Tags::ConstitutiveRelationPerBlock<Dim>::create_from_options(
          domain_creator, material_layers_variant);
  auto material_block_groups =
      Tags::MaterialBlockGroups<Dim>::create_from_options(
          material_layers_variant);
  REQUIRE(material_block_groups ==
          std::unordered_set<std::string>{"Layer0", "Layer1"});

  using element_array = ElementArray<Metavariables>;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      tuples::TaggedTuple<domain::Tags::Domain<Dim>,
                          Tags::ConstitutiveRelationPerBlock<Dim>,
                          Tags::MaterialBlockGroups<Dim>>{
          domain_creator->create_domain(), std::move(material_per_block),
          std::move(material_block_groups)}};
  for (const auto& element_id : {element_id_layer1, element_id_layer2}) {
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id, {Element<Dim>{element_id, {}}});
  }
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  for (const auto& element_id : {element_id_layer1, element_id_layer2}) {
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
  }

  const auto get_tag =
      [&runner](auto tag_v, const ElementId<Dim>& element_id) -> const auto& {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                              element_id);
  };

  const auto check_material = [&get_tag](const ElementId<Dim>& element_id,
                                         const double expected_bulk_modulus) {
    const auto& material =
        dynamic_cast<const ConstitutiveRelations::IsotropicHomogeneous<Dim>&>(
            get_tag(Tags::ConstitutiveRelation<Dim>{}, element_id));
    CHECK(material.bulk_modulus() == expected_bulk_modulus);
  };

  check_material(element_id_layer1, 1.);
  check_material(element_id_layer2, 3.);
}
}  // namespace Elasticity
