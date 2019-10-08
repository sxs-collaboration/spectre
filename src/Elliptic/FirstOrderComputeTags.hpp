// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Elliptic/Tags.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {
namespace Tags {

template <size_t Dim, typename System,
          typename VarsTag = typename System::variables_tag,
          typename FluxesComputer = typename System::fluxes,
          typename PrimalVars = typename System::primal_variables,
          typename AuxiliaryVars = typename System::auxiliary_variables,
          typename FluxesArgs = typename FluxesComputer::argument_tags>
struct FirstOrderFluxesCompute;

template <size_t Dim, typename System, typename VarsTag,
          typename FluxesComputer, typename... PrimalVars,
          typename... AuxiliaryVars, typename... FluxesArgs>
struct FirstOrderFluxesCompute<
    Dim, System, VarsTag, FluxesComputer, tmpl::list<PrimalVars...>,
    tmpl::list<AuxiliaryVars...>, tmpl::list<FluxesArgs...>>
    : db::add_tag_prefix<::Tags::Flux, VarsTag, tmpl::size_t<Dim>,
                         Frame::Inertial>,
      db::ComputeTag {
 private:
  using fluxes_computer_tag = elliptic::Tags::FluxesComputer<FluxesComputer>;

 public:
  using base = db::add_tag_prefix<::Tags::Flux, VarsTag, tmpl::size_t<Dim>,
                                  Frame::Inertial>;
  using argument_tags = tmpl::list<VarsTag, fluxes_computer_tag, FluxesArgs...>;
  using volume_tags = tmpl::list<fluxes_computer_tag>;
  static constexpr auto function(
      const db::const_item_type<VarsTag>& vars,
      const FluxesComputer& fluxes_computer,
      const db::const_item_type<FluxesArgs>&... fluxes_args) noexcept {
    auto fluxes = make_with_value<db::item_type<base>>(vars, 0.);
    // Compute fluxes for primal fields
    fluxes_computer.apply(
        make_not_null(
            &get<::Tags::Flux<PrimalVars, tmpl::size_t<Dim>, Frame::Inertial>>(
                fluxes))...,
        fluxes_args..., get<AuxiliaryVars>(vars)...);
    // Compute fluxes for auxiliary fields
    fluxes_computer.apply(
        make_not_null(&get<::Tags::Flux<AuxiliaryVars, tmpl::size_t<Dim>,
                                        Frame::Inertial>>(fluxes))...,
        fluxes_args..., get<PrimalVars>(vars)...);
    return fluxes;
  }
};

template <typename System, typename VarsTag = typename System::variables_tag,
          typename SourcesComputer = typename System::sources,
          typename PrimalVars = typename System::primal_variables,
          typename AuxiliaryVars = typename System::auxiliary_variables,
          typename SourcesArgs = typename SourcesComputer::argument_tags>
struct FirstOrderSourcesCompute;

template <typename System, typename VarsTag, typename SourcesComputer,
          typename... PrimalVars, typename... AuxiliaryVars,
          typename... SourcesArgs>
struct FirstOrderSourcesCompute<
    System, VarsTag, SourcesComputer, tmpl::list<PrimalVars...>,
    tmpl::list<AuxiliaryVars...>, tmpl::list<SourcesArgs...>>
    : db::add_tag_prefix<::Tags::Source, VarsTag>, db::ComputeTag {
  using base = db::add_tag_prefix<::Tags::Source, VarsTag>;
  using argument_tags = tmpl::list<VarsTag, SourcesArgs...>;
  static constexpr auto function(
      const db::const_item_type<VarsTag>& vars,
      const db::const_item_type<SourcesArgs>&... sources_args) noexcept {
    auto sources = make_with_value<db::item_type<base>>(vars, 0.);
    // Compute sources for primal fields
    SourcesComputer::apply(
        make_not_null(&get<::Tags::Source<PrimalVars>>(sources))...,
        sources_args..., get<PrimalVars>(vars)...);
    // Compute sources for auxiliary fields. They are just the auxiliary field
    // values.
    tmpl::for_each<tmpl::list<AuxiliaryVars...>>(
        [&sources, &vars ](const auto auxiliary_field_tag_v) noexcept {
          using auxiliary_field_tag =
              tmpl::type_from<decltype(auxiliary_field_tag_v)>;
          get<::Tags::Source<auxiliary_field_tag>>(sources) =
              get<auxiliary_field_tag>(vars);
        });
    return sources;
  }
};

}  // namespace Tags
}  // namespace elliptic
