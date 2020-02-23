// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "Elliptic/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {
namespace Tags {

template <typename System, typename VarsTag = typename System::variables_tag,
          typename FluxesComputer = typename System::fluxes,
          typename PrimalVars = typename System::primal_variables,
          typename AuxiliaryVars = typename System::auxiliary_variables,
          typename FluxesArgs = typename FluxesComputer::argument_tags>
struct FirstOrderFluxesCompute;

template <typename System, typename VarsTag, typename FluxesComputer,
          typename PrimalVars, typename AuxiliaryVars, typename... FluxesArgs>
struct FirstOrderFluxesCompute<System, VarsTag, FluxesComputer, PrimalVars,
                               AuxiliaryVars, tmpl::list<FluxesArgs...>>
    : db::add_tag_prefix<::Tags::Flux, VarsTag,
                         tmpl::size_t<System::volume_dim>, Frame::Inertial>,
      db::ComputeTag {
 private:
  static constexpr size_t volume_dim = System::volume_dim;
  using fluxes_computer_tag = elliptic::Tags::FluxesComputer<FluxesComputer>;

 public:
  using base = db::add_tag_prefix<::Tags::Flux, VarsTag,
                                  tmpl::size_t<volume_dim>, Frame::Inertial>;
  using argument_tags = tmpl::list<VarsTag, fluxes_computer_tag, FluxesArgs...>;
  using volume_tags = tmpl::list<fluxes_computer_tag>;
  static constexpr db::item_type<base> (*function)(
      const db::const_item_type<VarsTag>&, const FluxesComputer&,
      const db::const_item_type<FluxesArgs>&...) =
      &elliptic::first_order_fluxes<volume_dim, PrimalVars, AuxiliaryVars,
                                    db::get_variables_tags_list<VarsTag>,
                                    FluxesComputer,
                                    db::const_item_type<FluxesArgs>...>;
};

template <typename System, typename VarsTag = typename System::variables_tag,
          typename SourcesComputer = typename System::sources,
          typename PrimalVars = typename System::primal_variables,
          typename AuxiliaryVars = typename System::auxiliary_variables,
          typename SourcesArgs = typename SourcesComputer::argument_tags>
struct FirstOrderSourcesCompute;

template <typename System, typename VarsTag, typename SourcesComputer,
          typename PrimalVars, typename AuxiliaryVars, typename... SourcesArgs>
struct FirstOrderSourcesCompute<System, VarsTag, SourcesComputer, PrimalVars,
                                AuxiliaryVars, tmpl::list<SourcesArgs...>>
    : db::add_tag_prefix<::Tags::Source, VarsTag>, db::ComputeTag {
  using base = db::add_tag_prefix<::Tags::Source, VarsTag>;
  using argument_tags = tmpl::list<VarsTag, SourcesArgs...>;
  static constexpr db::item_type<base> (*function)(
      const db::const_item_type<VarsTag>&,
      const db::const_item_type<SourcesArgs>&...) =
      &elliptic::first_order_sources<PrimalVars, AuxiliaryVars, SourcesComputer,
                                     db::get_variables_tags_list<VarsTag>,
                                     db::const_item_type<SourcesArgs>...>;
};

}  // namespace Tags
}  // namespace elliptic
