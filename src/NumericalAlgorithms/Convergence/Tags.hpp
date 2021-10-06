// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "Options/Options.hpp"

namespace Convergence {
/// Option tags related to the convergence of iterative algorithms
namespace OptionTags {

template <typename OptionsGroup>
struct Criteria {
  static std::string name() { return "ConvergenceCriteria"; }
  static constexpr Options::String help =
      "Determine convergence of the algorithm";
  using type = Convergence::Criteria;
  using group = OptionsGroup;
};

template <typename OptionsGroup>
struct Iterations {
  static constexpr Options::String help =
      "Number of iterations to run the algorithm";
  using type = size_t;
  using group = OptionsGroup;
};

}  // namespace OptionTags

/// Tags related to the convergence of iterative algorithms
namespace Tags {

/// `Convergence::Criteria` that determine the iterative algorithm has converged
template <typename OptionsGroup>
struct Criteria : db::SimpleTag {
  static std::string name() {
    return "ConvergenceCriteria(" + Options::name<OptionsGroup>() + ")";
  }
  using type = Convergence::Criteria;

  using option_tags = tmpl::list<OptionTags::Criteria<OptionsGroup>>;
  static constexpr bool pass_metavariables = false;
  static Convergence::Criteria create_from_options(
      const Convergence::Criteria& convergence_criteria) {
    return convergence_criteria;
  }
};

/// A fixed number of iterations to run the iterative algorithm
template <typename OptionsGroup>
struct Iterations : db::SimpleTag {
  static std::string name() {
    return "Iterations(" + Options::name<OptionsGroup>() + ")";
  }
  using type = size_t;

  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<OptionTags::Iterations<OptionsGroup>>;
  static size_t create_from_options(const size_t max_iterations) {
    return max_iterations;
  }
};

/// Identifies a step in an iterative algorithm
template <typename Label>
struct IterationId : db::SimpleTag {
  static std::string name() {
    return "IterationId(" + Options::name<Label>() + ")";
  }
  using type = size_t;
};

/*!
 * \brief Holds a `Convergence::HasConverged` flag that signals the iterative
 * algorithm has converged, along with the reason for convergence.
 */
template <typename Label>
struct HasConverged : db::SimpleTag {
  static std::string name() {
    return "HasConverged(" + Options::name<Label>() + ")";
  }
  using type = Convergence::HasConverged;
};

}  // namespace Tags
}  // namespace Convergence
