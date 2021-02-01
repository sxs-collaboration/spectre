// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Helper functions to test elliptic first-order systems

#pragma once

#include <cstddef>
#include <random>
#include <tuple>
#include <limits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Elliptic/FirstOrderComputeTags.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "Framework/TestHelpers.hpp"
#include "Framework/TestingFramework.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace TestHelpers {
/// \ingroup TestingFrameworkGroup
/// Helper functions to test elliptic first-order systems
namespace elliptic {

/*!
 * \brief Test the `System::fluxes_computer` is functional
 *
 * This function tests the following properties of the
 * `System::fluxes_computer`:
 *
 * - It works with the `elliptic::first_order_fluxes` function.
 * - It can be applied to a DataBox, i.e. its argument tags are consistent with
 *   its apply function.
 */
template <typename System>
void test_first_order_fluxes_computer(
    const typename System::fluxes_computer& fluxes_computer,
    const DataVector& used_for_size) {
  using FluxesComputer = typename System::fluxes_computer;
  static constexpr size_t volume_dim = System::volume_dim;
  using vars_tag = typename System::fields_tag;
  using primal_fields = typename System::primal_fields;
  using auxiliary_fields = typename System::auxiliary_fields;
  using VarsType = typename vars_tag::type;
  using fluxes_tag =
      db::add_tag_prefix<::Tags::Flux, vars_tag, tmpl::size_t<volume_dim>,
                         Frame::Inertial>;
  using argument_tags = typename FluxesComputer::argument_tags;

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(0.5, 2.);

  // Generate random variables
  const auto vars = make_with_random_values<VarsType>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);

  // Generate fluxes from the variables with random arguments
  tuples::tagged_tuple_from_typelist<argument_tags> args{};
  tmpl::for_each<argument_tags>(
      [&args, &generator, &dist, &used_for_size](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        get<tag>(args) = make_with_random_values<typename tag::type>(
            make_not_null(&generator), make_not_null(&dist), used_for_size);
      });
  const auto expected_fluxes = tuples::apply(
      [&vars, &fluxes_computer](const auto&... expanded_args) {
        return ::elliptic::first_order_fluxes<volume_dim, primal_fields,
                                              auxiliary_fields>(
            vars, fluxes_computer, expanded_args...);
      },
      args);

  // Create a DataBox
  auto box = tuples::apply(
      [&vars, &used_for_size](const auto&... expanded_args) {
        return db::create<tmpl::append<db::AddSimpleTags<vars_tag, fluxes_tag>,
                                       argument_tags>>(
            vars,
            make_with_value<typename fluxes_tag::type>(
                used_for_size, std::numeric_limits<double>::signaling_NaN()),
            expanded_args...);
      },
      args);

  // Apply the fluxes computer to the DataBox
  db::mutate_apply<db::wrap_tags_in<::Tags::Flux, primal_fields,
                                    tmpl::size_t<volume_dim>, Frame::Inertial>,
                   tmpl::append<argument_tags, auxiliary_fields>>(
      fluxes_computer, make_not_null(&box));
  db::mutate_apply<db::wrap_tags_in<::Tags::Flux, auxiliary_fields,
                                    tmpl::size_t<volume_dim>, Frame::Inertial>,
                   tmpl::append<argument_tags, primal_fields>>(
      fluxes_computer, make_not_null(&box));
  CHECK(expected_fluxes == get<fluxes_tag>(box));
}

/*!
 * \brief Test the `System::sources_computer` is functional
 *
 * This function tests the following properties of the
 * `System::sources_computer`:
 *
 * - It works with the `elliptic::first_order_sources` function.
 * - It can be applied to a DataBox, i.e. its argument tags are consistent with
 *   its apply function.
 */
template <typename System>
void test_first_order_sources_computer(const DataVector& used_for_size) {
  using SourcesComputer = typename System::sources_computer;
  static constexpr size_t volume_dim = System::volume_dim;
  using vars_tag = typename System::fields_tag;
  using primal_fields = typename System::primal_fields;
  using auxiliary_fields = typename System::auxiliary_fields;
  using VarsType = typename vars_tag::type;
  using fluxes_tag =
      db::add_tag_prefix<::Tags::Flux, vars_tag, tmpl::size_t<volume_dim>,
                         Frame::Inertial>;
  using FluxesType = typename fluxes_tag::type;
  using sources_tag = db::add_tag_prefix<::Tags::Source, vars_tag>;
  using argument_tags = typename SourcesComputer::argument_tags;

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);

  // Generate random variables and fluxes. Note that the sources make use of the
  // pre-computed fluxes as an optimization (see elliptic::first_order_sources),
  // but for the purpose of this random-value test the fluxes don't have to be
  // computed from the variables.
  const auto vars = make_with_random_values<VarsType>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto fluxes = make_with_random_values<FluxesType>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);

  // Generate sources from the variables with random arguments
  tuples::tagged_tuple_from_typelist<argument_tags> args{};
  tmpl::for_each<argument_tags>(
      [&args, &generator, &dist, &used_for_size](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        get<tag>(args) = make_with_random_values<typename tag::type>(
            make_not_null(&generator), make_not_null(&dist), used_for_size);
      });
  const auto expected_sources = tuples::apply(
      [&vars, &fluxes](const auto&... expanded_args) {
        return ::elliptic::first_order_sources<
            volume_dim, primal_fields, auxiliary_fields, SourcesComputer>(
            vars, fluxes, expanded_args...);
      },
      args);

  // Create a DataBox
  auto box = tuples::apply(
      [&vars, &fluxes, &used_for_size](const auto&... expanded_args) {
        return db::create<
            tmpl::append<db::AddSimpleTags<vars_tag, fluxes_tag, sources_tag>,
                         argument_tags>>(
            vars, fluxes,
            make_with_value<typename sources_tag::type>(
                used_for_size, std::numeric_limits<double>::signaling_NaN()),
            expanded_args...);
      },
      args);
  tmpl::for_each<auxiliary_fields>([&vars, &box](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    db::mutate<::Tags::Source<tag>>(
        make_not_null(&box),
        [&vars](const auto aux_source) { *aux_source = get<tag>(vars); });
  });

  // Apply the sources computer to the DataBox
  db::mutate_apply<
      db::wrap_tags_in<::Tags::Source, typename vars_tag::tags_list>,
      tmpl::append<
          argument_tags, primal_fields,
          db::wrap_tags_in<::Tags::Flux, primal_fields,
                           tmpl::size_t<volume_dim>, Frame::Inertial>>>(
      SourcesComputer{}, make_not_null(&box));
  CHECK(expected_sources == get<sources_tag>(box));
}

}  // namespace elliptic
}  // namespace TestHelpers
