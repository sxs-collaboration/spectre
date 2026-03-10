// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <pup.h>
#include <string>
#include <tuple>
#include <unordered_set>

#include "DataStructures/ApplyMatrices.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filter.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
class Matrix;
template <size_t Dim>
class Mesh;

/// \endcond

namespace Filters {

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief A cached exponential filter.
 *
 * Applies an exponential filter in each logical direction to each component of
 * the tensors `TagsToFilter`. The exponential filter rescales the 1d modal
 * coefficients \f$c_i\f$ as:
 *
 * \f{align*}{
 *  c_i\to c_i \exp\left[-\alpha_{\mathrm{ef}}
 *   \left(\frac{i}{N}\right)^{2\beta_{\mathrm{ef}}}\right]
 * \f}
 *
 * where \f$N\f$ is the basis degree (number of grid points per element per
 * dimension minus one), \f$\alpha_{\mathrm{ef}}\f$ determines how much the
 * coefficients are rescaled, and \f$\beta_{\mathrm{ef}}\f$ (given by the
 * `HalfPower` option) determines how aggressive/broad the filter is (lower
 * values means filtering more coefficients). Setting
 * \f$\alpha_{\mathrm{ef}}=36\f$ results in effectively zeroing the highest
 * coefficient (in practice it gets rescaled by machine epsilon). The same
 * \f$\alpha_{\mathrm{ef}}\f$ and \f$\beta_{\mathrm{ef}}\f$ are used in each
 * logical direction. For a discussion of filtering see section 5.3 of
 * \cite HesthavenWarburton.
 *
 * This filter is skipped for mesh dimensions with basis `SphericalHarmonic`,
 * since Ylm filtering should be done for those. However, radial filtering is
 * done in this case (and can be disabled by specifying the blocks to filter).
 */
template <size_t Dim>
class Exponential : public Filter {
 public:
  /// \brief The value of `exp(-alpha)` is what the highest modal coefficient is
  /// rescaled by.
  struct Alpha {
    using type = double;
    static constexpr Options::String help =
        "exp(-alpha) is rescaling of highest coefficient";
    static type lower_bound() { return 0.0; }
  };

  /*!
   * \brief Half of the exponent in the exponential.
   *
   * \f{align*}{
   *  c_i\to c_i \exp\left[-\alpha \left(\frac{i}{N}\right)^{2m}\right]
   * \f}
   */
  struct HalfPower {
    using type = unsigned;
    static constexpr Options::String help =
        "Half of the exponent in the generalized Gaussian";
    static type lower_bound() { return 1; }
  };

  struct BlocksToFilter {
    using type =
        Options::Auto<std::vector<std::string>, Options::AutoLabel::All>;
    static constexpr Options::String help = {
        "List of blocks or block groups to apply filtering to. All other "
        "blocks will have no filtering. You can also specify 'All' to do "
        "filtering in all blocks of the domain."};
  };

  using options = tmpl::list<Alpha, HalfPower, BlocksToFilter>;
  static constexpr Options::String help = {"An exponential filter."};
  static std::string name() { return "ExponentialFilter"; }

  Exponential() = default;

  Exponential(double alpha, unsigned half_power,
              const std::optional<std::vector<std::string>>& blocks_to_filter,
              const Options::Context& context = {});

  WRAPPED_PUPable_decl_template(Exponential);  // NOLINT
  explicit Exponential(CkMigrateMessage* msg) : Filter(msg) {}

  /// A cached matrix used to apply the filter to the given mesh
  const Matrix& filter_matrix(const Mesh<1>& mesh) const;

  std::optional<std::unordered_set<std::string>> blocks_to_filter()
      const override {
    return blocks_to_filter_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 public:
  using argument_tags = tmpl::list<domain::Tags::Mesh<Dim>>;

  template <typename TagsList>
  void operator()(const gsl::not_null<Variables<TagsList>*> vars,
                  const Mesh<Dim>& mesh) const {
    *vars = apply_matrices(filter_matrices(mesh), *vars, mesh.extents());
  }

  template <typename... TensorTypes>
    requires((not tt::is_a_v<Variables, std::decay_t<TensorTypes>>) and ...)
  void operator()(const std::tuple<gsl::not_null<TensorTypes*>...>& tensors,
                  const Mesh<Dim>& mesh) const {
    const auto filter = filter_matrices(mesh);
    std::apply(
        [&filter, extents = mesh.extents()](const auto... tensor_ptrs) {
          (
              [&filter, &extents](const auto tensor_ptr) {
                for (auto& component : *tensor_ptr) {
                  component = apply_matrices(filter, component, extents);
                }
              }(tensor_ptrs),
              ...);
        },
        tensors);
  }

 private:
  std::array<std::reference_wrapper<const Matrix>, Dim> filter_matrices(
      const Mesh<Dim>& mesh) const;

  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const Exponential<LocalDim>& lhs,
                         const Exponential<LocalDim>& rhs);

  double alpha_{36.0};
  unsigned half_power_{16};
  std::optional<std::unordered_set<std::string>> blocks_to_filter_{};
};

template <size_t LocalDim>
bool operator==(const Exponential<LocalDim>& lhs,
                const Exponential<LocalDim>& rhs);

template <size_t LocalDim>
bool operator!=(const Exponential<LocalDim>& lhs,
                const Exponential<LocalDim>& rhs);

/// \cond
template <size_t Dim>
PUP::able::PUP_ID Exponential<Dim>::my_PUP_ID = 0;  // NOLINT
/// \endcond

}  // namespace Filters
