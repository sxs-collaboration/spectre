// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/SubitemTag.hpp"
#include "DataStructures/DataBox/Subitems.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Variables.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace OptionTags {
/// \ingroup OptionGroupsGroup
/// Holds the `OptionTags::AnalyticSolution` option in the input file
struct AnalyticSolutionGroup {
  static std::string name() { return "AnalyticSolution"; }
  static constexpr Options::String help =
      "Analytic solution used for the initial data and errors";
};

/// \ingroup OptionTagsGroup
/// The analytic solution, with the type of the analytic solution set as the
/// template parameter
template <typename SolutionType>
struct AnalyticSolution {
  static std::string name() { return pretty_type::name<SolutionType>(); }
  static constexpr Options::String help = "Options for the analytic solution";
  using type = SolutionType;
  using group = AnalyticSolutionGroup;
};
}  // namespace OptionTags

namespace Tags {
/// Can be used to retrieve the analytic solution from the cache without having
/// to know the template parameters of AnalyticSolution.
struct AnalyticSolutionBase : AnalyticSolutionOrData {};

/// \ingroup OptionTagsGroup
/// The analytic solution, with the type of the analytic solution set as the
/// template parameter
template <typename SolutionType>
struct AnalyticSolution : AnalyticSolutionBase, db::SimpleTag {
  using type = SolutionType;
  using option_tags = tmpl::list<::OptionTags::AnalyticSolution<SolutionType>>;

  static constexpr bool pass_metavariables = false;
  static SolutionType create_from_options(
      const SolutionType& analytic_solution) {
    return deserialize<type>(serialize<type>(analytic_solution).data());
  }
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating the analytic solution value for a quantity
///
/// \snippet AnalyticSolutions/Test_Tags.cpp analytic_name
template <typename Tag>
struct Analytic : db::PrefixTag, db::SimpleTag {
  using type = std::optional<typename Tag::type>;
  using tag = Tag;
};

/// Base tag for the analytic solution tensors.
///
/// \see ::Tags::AnalyticSolutions
struct AnalyticSolutionsBase : db::BaseTag {};

namespace detail {
template <typename Tag>
struct AnalyticImpl : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
}  // namespace detail

/*!
 * \brief The analytic solution of the `FieldTags`.
 *
 * The `std::optional` is a `nullopt` if there is no analytic solution.
 *
 * The individual `FieldTags` are added as `Tags::Analytic<Tag>` which holds a
 * `std::optional<Tensor>`.
 */
template <typename FieldTags>
struct AnalyticSolutions : ::Tags::AnalyticSolutionsBase, db::SimpleTag {
  using field_tags = FieldTags;
  using type = std::optional<
      ::Variables<db::wrap_tags_in<detail::AnalyticImpl, FieldTags>>>;
};

namespace detail {
// Check if the argument is a `::Tags::AnalyticSolutions` template, or derived
// from it
template <typename FieldTags>
constexpr std::true_type is_analytic_solutions(
    ::Tags::AnalyticSolutions<FieldTags>&&) {
  return {};
}

constexpr std::false_type is_analytic_solutions(...) { return {}; }

template <typename Tag>
static constexpr bool is_analytic_solutions_v =
    decltype(is_analytic_solutions(std::declval<Tag>()))::value;
}  // namespace detail

/*!
 * \brief The error of the `Tag` defined as `numerical - analytic`.
 *
 * The `std::optional` is a `nullopt` if no error was able to be computed, e.g.
 * if there is no analytic solution to compare to.
 */
template <typename Tag>
struct Error : db::PrefixTag, db::SimpleTag {
  using type = std::optional<typename Tag::type>;
  using tag = Tag;
};

namespace detail {
template <typename Tag>
struct ErrorImpl : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
}  // namespace detail

/*!
 * \brief The error of the `FieldTags`, defined as `numerical - analytic`.
 *
 * The `std::optional` is a `nullopt` if no error was able to be computed, e.g.
 * if there is no analytic solution to compare to.
 *
 * The individual `FieldTags` are added as `Tags::Error<Tag>` which holds a
 * `std::optional<Tensor>`.
 */
template <typename FieldTags>
struct Errors : db::SimpleTag {
  using field_tags = FieldTags;
  using type = std::optional<
      ::Variables<db::wrap_tags_in<detail::ErrorImpl, FieldTags>>>;
};

/// \cond
template <typename FieldTagsList>
struct ErrorsCompute;
/// \endcond

/*!
 * \brief Compute tag for computing the error from the `Tags::Analytic` of the
 * `FieldTags`.
 *
 * The error is defined as `numerical - analytic`.
 *
 * We use individual `Tensor`s rather than `Variables` of the `FieldTags`
 * because not all `FieldTags` are always stored in the same `Variables`.
 * For example, in the Generalized Harmonic-GRMHD combined system, the analytic
 * variables for the GH system are part of the `evolved_variables_tag` while the
 * GRMHD analytic variables are part of the `primitive_variables_tag`. A similar
 * issue arises in some elliptic systems. The main drawback of having to use the
 * tensor-by-tensor implementation instead of a `Variables` implementation is
 * the added loop complexity. However, this is offset by the reduced code
 * duplication and flexibility.
 */
template <typename... FieldTags>
struct ErrorsCompute<tmpl::list<FieldTags...>>
    : Errors<tmpl::list<FieldTags...>>, db::ComputeTag {
  using field_tags = tmpl::list<FieldTags...>;
  using base = Errors<tmpl::list<FieldTags...>>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<FieldTags..., ::Tags::Analytic<FieldTags>...>;
  static void function(
      const gsl::not_null<return_type*> errors,
      const typename FieldTags::type&... vars,
      const typename ::Tags::Analytic<FieldTags>::type&... analytic_vars) {
    if (const auto& first_analytic_var = get_first_argument(analytic_vars...);
        first_analytic_var.has_value()) {
      // Construct a Variables of the size of the DataVector of the first
      // analytic_vars tensor.
      *errors = typename return_type::value_type{
          first_analytic_var.value()[0].size()};
      const auto compute_error = [](const auto error_ptr, const auto& var,
                                    const auto& analytic_var) {
        for (size_t tensor_index = 0; tensor_index < var.size();
             ++tensor_index) {
          (*error_ptr)[tensor_index] =
              var[tensor_index] - analytic_var[tensor_index];
        }
      };
      EXPAND_PACK_LEFT_TO_RIGHT(compute_error(
          make_not_null(
              &get<::Tags::detail::ErrorImpl<FieldTags>>(errors->value())),
          vars, analytic_vars.value()));
    } else {
      *errors = std::nullopt;
    }
  }
};

namespace detail {
// Check if the argument is a `::Tags::Errors` template, or derived
// from it
template <typename FieldTags>
constexpr std::true_type is_errors(::Tags::Errors<FieldTags>&&) {
  return {};
}

constexpr std::false_type is_errors(...) { return {}; }

template <typename Tag>
static constexpr bool is_errors_v =
    decltype(is_errors(std::declval<Tag>()))::value;
}  // namespace detail
}  // namespace Tags

/// \cond
namespace Tags {
template <typename Tag, typename ParentTag>
struct Subitem<Tag, ParentTag,
               Requires<detail::is_analytic_solutions_v<ParentTag>>>
    : Tag, db::ComputeTag {
  using field_tags = typename ParentTag::field_tags;
  using base = Tag;
  using parent_tag = AnalyticSolutions<field_tags>;
  using argument_tags = tmpl::list<parent_tag>;
  static void function(const gsl::not_null<typename base::type*> result,
                       const typename parent_tag::type& vars) {
    if (vars.has_value()) {
      result->reset();
      *result = typename base::type::value_type{};
      const auto& tensor =
          get<typename detail::AnalyticImpl<typename Tag::tag>>(*vars);
      const size_t num_components = tensor.size();
      for (size_t storage_index = 0; storage_index < num_components;
           ++storage_index) {
        (**result)[storage_index].set_data_ref(
            make_not_null(&const_cast<DataVector&>(tensor[storage_index])));
      }
    } else {
      *result = std::nullopt;
    }
  }
};

template <typename Tag, typename ParentTag>
struct Subitem<Tag, ParentTag, Requires<detail::is_errors_v<ParentTag>>>
    : Tag, db::ComputeTag {
  using field_tags = typename ParentTag::field_tags;
  using base = Tag;
  using parent_tag = Errors<field_tags>;
  using argument_tags = tmpl::list<parent_tag>;
  static void function(const gsl::not_null<typename base::type*> result,
                       const typename parent_tag::type& vars) {
    if (vars.has_value()) {
      result->reset();
      *result = typename base::type::value_type{};
      const auto& tensor =
          get<typename detail::ErrorImpl<typename Tag::tag>>(*vars);
      const size_t num_components = tensor.size();
      for (size_t storage_index = 0; storage_index < num_components;
           ++storage_index) {
        (**result)[storage_index].set_data_ref(
            make_not_null(&const_cast<DataVector&>(tensor[storage_index])));
      }
    } else {
      *result = std::nullopt;
    }
  }
};
}  // namespace Tags

namespace db {
template <typename Tag>
struct Subitems<Tag, Requires<Tags::detail::is_analytic_solutions_v<Tag>>> {
  using field_tags = typename Tag::field_tags;
  using type = db::wrap_tags_in<::Tags::Analytic, field_tags>;

  template <typename Subtag>
  static void create_item(
      const gsl::not_null<typename Tag::type*> parent_value,
      const gsl::not_null<typename Subtag::type*> sub_value) {
    if (parent_value->has_value()) {
      auto& tensor = get<::Tags::detail::AnalyticImpl<typename Subtag::tag>>(
          parent_value->value());
      sub_value->emplace();
      // Only update the Tensor if the Variables has changed its allocation
      if constexpr (not is_any_spin_weighted_v<
                        typename Subtag::tag::type::type>) {
        if (tensor.begin()->data() != sub_value->value().begin()->data()) {
          for (auto tensor_component = tensor.begin(),
                    sub_var_it = sub_value->value().begin();
               tensor_component != tensor.end();
               ++tensor_component, ++sub_var_it) {
            sub_var_it->set_data_ref(make_not_null(&*tensor_component));
          }
        }
      } else {
        if (tensor.begin()->data().data() !=
            sub_value->value().begin()->data().data()) {
          for (auto tensor_component = tensor.begin(),
                    sub_var_it = sub_value->value().begin();
               tensor_component != tensor.end();
               ++tensor_component, ++sub_var_it) {
            sub_var_it->set_data_ref(make_not_null(&*tensor_component));
          }
        }
      }
    } else {
      *sub_value = std::nullopt;
    }
  }
};

template <typename Tag>
struct Subitems<Tag, Requires<Tags::detail::is_errors_v<Tag>>> {
  using field_tags = typename Tag::field_tags;
  using type = db::wrap_tags_in<::Tags::Error, field_tags>;

  template <typename Subtag>
  static void create_item(
      const gsl::not_null<typename Tag::type*> parent_value,
      const gsl::not_null<typename Subtag::type*> sub_value) {
    if (parent_value->has_value()) {
      auto& tensor = get<::Tags::detail::ErrorImpl<typename Subtag::tag>>(
          parent_value->value());
      *sub_value = std::decay_t<decltype(tensor)>{};
      // Only update the Tensor if the Variables has changed its allocation
      if constexpr (not is_any_spin_weighted_v<
                        typename Subtag::tag::type::type>) {
        if (tensor.begin()->data() != sub_value->value().begin()->data()) {
          for (auto tensor_component = tensor.begin(),
                    sub_var_it = sub_value->value().begin();
               tensor_component != tensor.end();
               ++tensor_component, ++sub_var_it) {
            sub_var_it->set_data_ref(make_not_null(&*tensor_component));
          }
        }
      } else {
        if (tensor.begin()->data().data() !=
            sub_value->value().begin()->data().data()) {
          for (auto tensor_component = tensor.begin(),
                    sub_var_it = sub_value->value().begin();
               tensor_component != tensor.end();
               ++tensor_component, ++sub_var_it) {
            sub_var_it->set_data_ref(make_not_null(&*tensor_component));
          }
        }
      }
    } else {
      *sub_value = std::nullopt;
    }
  }
};
}  // namespace db
/// \endcond
