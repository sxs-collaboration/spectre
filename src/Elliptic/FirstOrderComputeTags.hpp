// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "Elliptic/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {
namespace Tags {

template <typename System>
struct FirstOrderFluxesCompute
    : db::add_tag_prefix<::Tags::Flux, typename System::variables_tag,
                         tmpl::size_t<System::volume_dim>, Frame::Inertial>,
      db::ComputeTag {
 private:
  static constexpr size_t volume_dim = System::volume_dim;
  using vars_tag = typename System::variables_tag;
  using FluxesComputer = typename System::fluxes;
  using fluxes_computer_tag = elliptic::Tags::FluxesComputer<FluxesComputer>;

 public:
  using base = db::add_tag_prefix<::Tags::Flux, vars_tag,
                                  tmpl::size_t<volume_dim>, Frame::Inertial>;
  using argument_tags = tmpl::push_front<typename FluxesComputer::argument_tags,
                                         vars_tag, fluxes_computer_tag>;
  using volume_tags =
      tmpl::push_front<get_volume_tags<FluxesComputer>, fluxes_computer_tag>;
  using return_type = db::item_type<base>;
  template <typename... FluxesArgs>
  static void function(const gsl::not_null<return_type*> fluxes,
                       const db::const_item_type<vars_tag>& vars,
                       const FluxesComputer& fluxes_computer,
                       const FluxesArgs&... fluxes_args) noexcept {
    *fluxes = return_type{vars.number_of_grid_points()};
    elliptic::first_order_fluxes<volume_dim, typename System::primal_variables,
                                 typename System::auxiliary_variables>(
        fluxes, vars, fluxes_computer, fluxes_args...);
  }
};

template <typename System>
struct FirstOrderSourcesCompute
    : db::add_tag_prefix<::Tags::Source, typename System::variables_tag>,
      db::ComputeTag {
 private:
  using vars_tag = typename System::variables_tag;
  using SourcesComputer = typename System::sources;

 public:
  using base = db::add_tag_prefix<::Tags::Source, vars_tag>;
  using argument_tags =
      tmpl::push_front<typename SourcesComputer::argument_tags, vars_tag>;
  using volume_tags = get_volume_tags<SourcesComputer>;
  using return_type = db::item_type<base>;
  template <typename... SourcesArgs>
  static void function(const gsl::not_null<return_type*> sources,
                       const db::const_item_type<vars_tag>& vars,
                       const SourcesArgs&... sources_args) noexcept {
    *sources = return_type{vars.number_of_grid_points()};
    elliptic::first_order_sources<typename System::primal_variables,
                                  typename System::auxiliary_variables,
                                  SourcesComputer>(sources, vars,
                                                   sources_args...);
  }
};

}  // namespace Tags
}  // namespace elliptic
