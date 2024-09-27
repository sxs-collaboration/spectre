// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <exception>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/ExpandOverBlocks.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Structure/BlockGroups.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Tags.hpp"
#include "IO/Observer/Tags.hpp"
#include "Options/String.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename... Tags>
struct TaggedTuple;
}  // namespace tuples
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

namespace Elasticity {
namespace Tags {

/// A constitutive relation in every block of the domain
template <size_t Dim>
struct ConstitutiveRelationPerBlock : db::SimpleTag,
                                      ConstitutiveRelationPerBlockBase {
  using ConstRelPtr =
      std::unique_ptr<ConstitutiveRelations::ConstitutiveRelation<Dim>>;
  using type = std::vector<ConstRelPtr>;

  using option_tags = tmpl::list<domain::OptionTags::DomainCreator<Dim>,
                                 OptionTags::ConstitutiveRelationPerBlock<Dim>>;
  static constexpr bool pass_metavariables = false;

  static type create_from_options(
      const std::unique_ptr<DomainCreator<Dim>>& domain_creator,
      const std::variant<ConstRelPtr, std::vector<ConstRelPtr>,
                         std::unordered_map<std::string, ConstRelPtr>>&
          constitutive_relation_per_block) {
    const auto block_names = domain_creator->block_names();
    const auto block_groups = domain_creator->block_groups();
    const domain::ExpandOverBlocks<ConstRelPtr> expand_over_blocks{
        block_names, block_groups};
    try {
      return std::visit(expand_over_blocks, constitutive_relation_per_block);
    } catch (const std::exception& error) {
      ERROR_NO_TRACE("Invalid 'Material':\n" << error.what());
    }
  }
};

/// References the constitutive relation for the element's block, which is
/// stored in the global cache
template <size_t Dim>
struct ConstitutiveRelationReference : ConstitutiveRelation<Dim>,
                                       db::ReferenceTag {
  using base = ConstitutiveRelation<Dim>;
  using argument_tags =
      tmpl::list<ConstitutiveRelationPerBlockBase, domain::Tags::Element<Dim>>;
  static const ConstitutiveRelations::ConstitutiveRelation<Dim>& get(
      const std::vector<
          std::unique_ptr<ConstitutiveRelations::ConstitutiveRelation<Dim>>>&
          constitutive_relation_per_block,
      const Element<Dim>& element) {
    return *constitutive_relation_per_block.at(element.id().block_id());
  }
};

/// Stores the names of the block groups that split the domain into layers with
/// different material properties. Useful to observe quantities in each layer.
template <size_t Dim>
struct MaterialBlockGroups : db::SimpleTag {
  using type = std::unordered_set<std::string>;

  using option_tags = tmpl::list<OptionTags::ConstitutiveRelationPerBlock<Dim>>;
  static constexpr bool pass_metavariables = false;
  using ConstRelPtr =
      std::unique_ptr<ConstitutiveRelations::ConstitutiveRelation<Dim>>;

  static type create_from_options(
      const std::variant<ConstRelPtr, std::vector<ConstRelPtr>,
                         std::unordered_map<std::string, ConstRelPtr>>&
          constitutive_relation_per_block) {
    if (std::holds_alternative<std::unordered_map<std::string, ConstRelPtr>>(
            constitutive_relation_per_block)) {
      const auto& map = std::get<std::unordered_map<std::string, ConstRelPtr>>(
          constitutive_relation_per_block);
      std::unordered_set<std::string> block_groups;
      for (const auto& [block_name, _] : map) {
        block_groups.insert(block_name);
      }
      return block_groups;
    } else {
      return {};
    }
  }
};

/// The name of the material layer (name of a block group with some material)
struct MaterialLayerName : db::SimpleTag {
  using type = std::optional<std::string>;
};

}  // namespace Tags

/// Actions related to solving Elasticity systems
namespace Actions {

/*!
 * \brief Initialize the constitutive relation describing properties of the
 * elastic material
 *
 * Every block in the domain can have a different constitutive relation,
 * allowing for composite materials. All constitutive relations are stored in
 * the global cache indexed by block, and elements reference their block's
 * constitutive relation in the DataBox. This means an element can retrieve the
 * local constitutive relation from the DataBox simply by requesting
 * `Elasticity::Tags::ConstitutiveRelation<Dim>`.
 */
template <size_t Dim>
struct InitializeConstitutiveRelation
    : tt::ConformsTo<amr::protocols::Projector> {
 public:  // Iterable action
  using const_global_cache_tags =
      tmpl::list<Tags::ConstitutiveRelationPerBlock<Dim>,
                 Tags::MaterialBlockGroups<Dim>>;
  using simple_tags =
      tmpl::list<Tags::MaterialLayerName,
                 observers::Tags::ObservationKey<Tags::MaterialLayerName>>;
  using compute_tags = tmpl::list<Tags::ConstitutiveRelationReference<Dim>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate_apply<InitializeConstitutiveRelation>(make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }

 public:  // amr::protocols::Projector
  using return_tags = simple_tags;
  using argument_tags =
      tmpl::list<Tags::MaterialBlockGroups<Dim>, domain::Tags::Element<Dim>,
                 domain::Tags::Domain<Dim>>;

  template <typename... AmrData>
  static void apply(
      const gsl::not_null<std::optional<std::string>*> material_layer_name,
      const gsl::not_null<std::optional<std::string>*> observation_key,
      const std::unordered_set<std::string>& material_block_groups,
      const Element<Dim>& element, const Domain<Dim>& domain,
      const AmrData&... /*unused*/) {
    const auto& block = domain.blocks()[element.id().block_id()];
    // Check if this element is in a material layer
    *material_layer_name = [&material_block_groups, &domain,
                            &block]() -> std::optional<std::string> {
      for (const auto& name : material_block_groups) {
        if (domain::block_is_in_group(block.name(), name,
                                      domain.block_groups())) {
          return name;
        }
      }
      return std::nullopt;
    }();
    // Set the corresponding observation key, but only on the finest multigrid
    // level. This could be done better by supporting intersections of array
    // sections in observation events or something like that.
    *observation_key =
        element.id().grid_index() == 0 ? *material_layer_name : std::nullopt;
  }
};

}  // namespace Actions
}  // namespace Elasticity
