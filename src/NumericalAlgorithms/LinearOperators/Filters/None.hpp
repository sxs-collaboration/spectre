// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
template <size_t Dim>
class Mesh;
/// \endcond

namespace Filters {
/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief A no-op filter that never modifies any data and accepts any mesh
 * basis or quadrature.
 *
 * See `Filters::Filter` for the general interface description.
 */
template <size_t Dim, typename TagList>
class None : public Filter<Dim, TagList> {
 public:
  /// \brief Which blocks the filter should be applied to.
  struct BlocksToFilter {
    using type =
        Options::Auto<std::vector<std::string>, Options::AutoLabel::All>;
    static constexpr Options::String help = {
        "List of blocks or block groups to restrict this no-op filter to. "
        "You can also specify 'All' to apply to every block in the domain. "
        "Since the filter never modifies data, this option only affects which "
        "blocks appear in the filter's block list."};
  };

  using options = tmpl::list<BlocksToFilter>;

  static constexpr Options::String help = {
      "A no-op filter that never modifies any data and is valid for any basis "
      "or quadrature."};

  None() = default;

  explicit None(const std::optional<std::vector<std::string>>& blocks_to_filter,
                const Options::Context& context = {});

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(Filter<Dim, TagList>), None);
  explicit None(CkMigrateMessage* msg) : Filter<Dim, TagList>(msg) {}

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  std::unique_ptr<Filter<Dim, TagList>> get_clone() const override {
    return std::make_unique<None>(*this);
  }

  bool apply_volume_filter_on_substep() const override { return false; }
  bool apply_volume_filter_on_this_step(size_t /*step_number*/) const override {
    return false;
  }

  bool apply_boundary_filter_on_substep() const override { return false; }
  bool apply_boundary_filter_on_this_step(
      size_t /*step_number*/) const override {
    return false;
  }

  bool need_jacobians() const override { return false; }

  bool supports_mesh(const Mesh<Dim>& /*mesh*/) const override { return true; }

  const std::optional<std::vector<size_t>>& blocks_to_filter() const override;

  void set_blocks_to_filter(
      const std::vector<std::string>& all_block_names,
      const std::unordered_map<std::string, std::unordered_set<std::string>>&
          block_groups) override;

  void apply_in_volume(
      gsl::not_null<Variables<TagList>*> vars, const Mesh<Dim>& mesh,
      const std::optional<
          InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>>&
          inv_jac_grid_to_inertial,
      const std::optional<
          Jacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>>&
          jac_grid_to_inertial) const override;

  void apply_on_boundary(
      gsl::not_null<Variables<TagList>*> vars, const Mesh<Dim - 1>& mesh,
      const std::optional<
          InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>>&
          inv_jac_grid_to_inertial,
      const std::optional<
          Jacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>>&
          jac_grid_to_inertial) const override;

  bool is_equal(const Filter<Dim, TagList>& other) const override;

 private:
  template <size_t LocalDim, typename LocalTagList>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const None<LocalDim, LocalTagList>& lhs,
                         const None<LocalDim, LocalTagList>& rhs);

  std::optional<std::vector<std::string>> blocks_and_groups_to_filter_{};
  std::optional<std::vector<size_t>> blocks_to_filter_{};
};

template <size_t Dim, typename TagList>
bool operator==(const None<Dim, TagList>& lhs, const None<Dim, TagList>& rhs);

template <size_t Dim, typename TagList>
bool operator!=(const None<Dim, TagList>& lhs, const None<Dim, TagList>& rhs);
}  // namespace Filters
