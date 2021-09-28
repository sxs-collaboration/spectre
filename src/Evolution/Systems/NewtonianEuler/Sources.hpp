// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler {

/*!
 * \brief Compute the source terms for the NewtonianEuler evolution
 * using a problem-specific source.
 *
 * \details The source term type is fetched from the initial data, so
 * any `InitialDataType` instance must have a type alias `source_term_type`.
 * In turn, any source term type used by `ComputeSources` must hold type aliases
 * `sourced_variables` and `argument_tags`, which are `tmpl::list`s of the
 * variables whose equations of motion require a source term, and the arguments
 * required to compute those source terms, respectively. The source term must
 * also hold a `public` `void` member function `apply` whose arguments are
 * `gsl::not_null` pointers to the variables storing the source terms, followed
 * by the arguments required to compute them.
 * See NewtonianEuler::Sources::UniformAcceleration for an example.
 *
 * While most of physically relevant source terms for the Newtonian Euler
 * equations do not add a source term for the mass density, this class allows
 * for problems that source any set of conserved variables
 * (at least one variable is required).
 */
template <typename InitialDataType>
struct ComputeSources {
 private:
  using source_term_type = typename InitialDataType::source_term_type;

  template <typename SourcedVarsTagList, typename ArgTagsList>
  struct apply_helper;

  template <typename... SourcedVarsTags, typename... ArgsTags>
  struct apply_helper<tmpl::list<SourcedVarsTags...>, tmpl::list<ArgsTags...>> {
    static void function(
        const gsl::not_null<typename SourcedVarsTags::type*>... sourced_vars,
        const typename Tags::SourceTerm<InitialDataType>::type& source,
        const typename ArgsTags::type&... args) {
      source.apply(sourced_vars..., args...);
    }
  };

 public:
  using return_tags =
      db::wrap_tags_in<::Tags::Source,
                       typename source_term_type::sourced_variables>;

  using argument_tags =
      tmpl::push_front<typename source_term_type::argument_tags,
                       Tags::SourceTerm<InitialDataType>>;

  template <class... Args>
  static void apply(const Args&... args) {
    apply_helper<typename source_term_type::sourced_variables,
                 typename source_term_type::argument_tags>::function(args...);
  }
};

}  // namespace NewtonianEuler
