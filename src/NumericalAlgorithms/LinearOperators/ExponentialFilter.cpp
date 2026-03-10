// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/ExponentialFilter.hpp"

#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_set>

#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Options.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace Filters {

template <size_t Dim>
Exponential<Dim>::Exponential(
    const double alpha, const unsigned half_power,
    const std::optional<std::vector<std::string>>& blocks_to_filter,
    const Options::Context& context)
    : alpha_(alpha), half_power_(half_power) {
  if (blocks_to_filter.has_value()) {
    blocks_to_filter_ = std::unordered_set<std::string>{};
    for (const std::string& block_name : blocks_to_filter.value()) {
      if (blocks_to_filter_->count(block_name) != 0) {
        PARSE_ERROR(context,
                    "Duplicate block name '"
                        << block_name
                        << "' found when creating an Exponential filter.");
      }

      blocks_to_filter_->emplace(block_name);
    }
  }
}

template <size_t Dim>
const Matrix& Exponential<Dim>::filter_matrix(const Mesh<1>& mesh) const {
  // We don't use StaticCache here because the matrices depend on additional
  // runtime parameters (alpha, half_power)
  using key_type = std::tuple<double, unsigned, size_t, Spectral::Basis,
                              Spectral::Quadrature>;
  static std::map<key_type, Matrix> cache{};
  static std::mutex cache_mutex{};

  const key_type key{alpha_, half_power_, mesh.extents(0), mesh.basis(0),
                     mesh.quadrature(0)};
  const std::lock_guard<std::mutex> lock(cache_mutex);
  auto [iter, inserted] = cache.emplace(key, Matrix{});
  if (inserted) {
    iter->second =
        Spectral::filtering::exponential_filter(mesh, alpha_, half_power_);
  }
  return iter->second;
}

template <size_t Dim>
std::array<std::reference_wrapper<const Matrix>, Dim>
Exponential<Dim>::filter_matrices(const Mesh<Dim>& mesh) const {
  const Matrix empty{};
  std::array<std::reference_wrapper<const Matrix>, Dim> filter =
      make_array<Dim>(std::cref(empty));
  for (size_t d = 0; d < Dim; d++) {
    gsl::at(filter, d) = std::cref(filter_matrix(mesh.slice_through(d)));
  }
  return filter;
}

template <size_t Dim>
void Exponential<Dim>::pup(PUP::er& p) {
  Filter::pup(p);
  p | alpha_;
  p | half_power_;
  p | blocks_to_filter_;
}

template <size_t LocalDim>
bool operator==(const Exponential<LocalDim>& lhs,
                const Exponential<LocalDim>& rhs) {
  return lhs.alpha_ == rhs.alpha_ and lhs.half_power_ == rhs.half_power_ and
         lhs.blocks_to_filter_ == rhs.blocks_to_filter_;
}

template <size_t LocalDim>
bool operator!=(const Exponential<LocalDim>& lhs,
                const Exponential<LocalDim>& rhs) {
  return not(lhs == rhs);
}

// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GEN_OP(op, dim)                                  \
  template bool operator op(const Exponential<dim>& lhs, \
                            const Exponential<dim>& rhs);
#define INSTANTIATE(_, data)             \
  template class Exponential<DIM(data)>; \
  GEN_OP(==, DIM(data))                  \
  GEN_OP(!=, DIM(data))

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef GEN_OP
#undef DIM

}  // namespace Filters
