// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup ConservativeGroup
/// \brief Calculate \f$\partial u/\partial t\f$ for a conservative system.
///
/// The time evolution of the variables \f$u\f$ of any conservative
/// system is given by \f$\partial_t u = - \partial_i F^i + S\f$,
/// where \f$F^i\f$ are the fluxes and \f$S\f$ are the sources.
///
/// Source terms are only added for variables in the
/// `System::sourced_variables` type list.
template <typename System>
struct ConservativeDuDt {
 private:
  template <typename TensorTagList, typename SourcedTagList>
  struct apply_helper;

  template <typename... TensorTags, typename... SourcedTags>
  struct apply_helper<tmpl::list<TensorTags...>, tmpl::list<SourcedTags...>> {
    static void function(
        const gsl::not_null<db::item_type<TensorTags>*>... dt_u,
        const db::item_type<TensorTags>&... div_flux,
        const db::item_type<SourcedTags>&... sources) noexcept {
      const auto negate = [](const auto a, const auto& b) noexcept {
        for (size_t i = 0; i < a->size(); ++i) {
          (*a)[i] = -b[i];
        }
        return nullptr;
      };
      expand_pack(negate(dt_u, div_flux)...);
      const auto add = [](const auto a, const auto& b) noexcept {
        for (size_t i = 0; i < a->size(); ++i) {
          (*a)[i] += b[i];
        }
        return nullptr;
      };
      (void)add;
      expand_pack(add(
          get<tmpl::index_of<tmpl::list<TensorTags...>, SourcedTags>::value>(
              std::forward_as_tuple(dt_u...)),
          sources)...);
    }
  };

 public:
  static constexpr size_t volume_dim = System::volume_dim;
  using frame = Frame::Inertial;

  using argument_tags = tmpl::append<
      db::split_tag<db::add_tag_prefix<
          Tags::div,
          db::add_tag_prefix<Tags::Flux, typename System::variables_tag,
                             tmpl::size_t<volume_dim>, frame>>>,
      tmpl::transform<typename System::sourced_variables,
                      tmpl::bind<Tags::Source, tmpl::_1>>>;

  static constexpr auto apply =
      apply_helper<db::split_tag<typename System::variables_tag>,
                   typename System::sourced_variables>::function;
};
