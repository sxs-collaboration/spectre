// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Element.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "Options/Options.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <typename T>
class DataVectorImpl;
using DataVector = DataVectorImpl<double>;
template <size_t VolumeDim>
class Direction;
template <size_t>
class Mesh;

namespace PUP {
class er;
}  // namespace PUP

namespace SlopeLimiters {
template <size_t VolumeDim, typename TagsToLimit>
class Minmod;
}  // namespace SlopeLimiters
/// \endcond

namespace SlopeLimiters {
/// \ingroup SlopeLimitersGroup
/// \brief Possible types of the minmod slope limiter.
///
/// \see SlopeLimiters::Minmod
enum class MinmodType { LambdaPi1, LambdaPiN, Muscl };
}  // namespace SlopeLimiters

namespace Minmod_detail {
// Encodes the return status of the minmod_tvbm function.
struct MinmodResult {
  const double value;
  const bool activated;
};

// The TVBM-corrected minmod function, see e.g. Cockburn reference Eq. 2.26.
MinmodResult minmod_tvbm(double a, double b, double c,
                         double tvbm_scale) noexcept;

// Implements the minmod limiter for one Tensor<DataVector>.
//
// The interface is designed to erase the tensor structure information, because
// this way the implementation can be moved out of the header file. This is
// achieved by receiving Tensor<DataVector>::iterators into the tensor to limit,
// and Tensor<double>::iterators into the neighbor tensors.
//
// Note: because the interface erases the tensor structure information, we can
// no longer rely on the compiler to enforce that the local and neighbor tensors
// share the same Structure.
template <size_t VolumeDim>
bool limit_one_tensor(
    gsl::not_null<DataVector*> tensor_begin,
    gsl::not_null<DataVector*> tensor_end,
    const std::unordered_map<Direction<VolumeDim>,
                             gsl::not_null<const double*>>&
        neighbor_tensor_begin,
    const SlopeLimiters::MinmodType& minmod_type, double tvbm_constant,
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
    const tnsr::I<DataVector, VolumeDim, Frame::Logical>& logical_coords,
    const tnsr::I<double, VolumeDim>& element_size,
    const std::unordered_map<Direction<VolumeDim>, tnsr::I<double, VolumeDim>>&
        neighbor_sizes) noexcept;

template <typename TensorType>
using tensor_double_from = Tensor<double, typename TensorType::symmetry,
                                  typename TensorType::index_list>;
}  // namespace Minmod_detail

namespace SlopeLimiters {
/// \ingroup SlopeLimitersGroup
/// \brief A generic Minmod slope limiter
///
/// Implements the minmod-based slope limiter from
/// \ref cockburn_ref "Cockburn (1999)", Section 2.4.
/// Three types of minmod limiter from the reference are implemented:
/// \f$\Lambda\Pi^1\f$, \f$\Lambda\Pi^N\f$, and MUSCL.
///
/// This minmod limiter has a generic implementation that can work on an
/// arbitrary set of tensors. The minmod limiting algorithm is applied to each
/// component of each tensor independently. In general, the limiter linearizes
/// the tensors on every DG element, each time it is applied; additionally, the
/// limiter may reduce the spatial slope of some tensor components if the data
/// look like they may contain oscillations.
///
/// The key features differentiating the three minmod limiter types are:
/// 1. The `Muscl` limiter is the most dissipative; it more aggressively reduces
///    the slopes of the data. This limiter may better handle strong shocks, but
///    also produces the most broadening of features.
/// 2. The `LambdaPiN` limiter is the least aggressive; its "troubled cell"
///    detector tries to avoid limiting in DG elements where the data look
///    smooth enough. Where `LambdaPiN` is able to avoid limiting, the data are
///    _not_ linearized, and the post-limiter data are identical to the
///    pre-limiter data.
/// 3. The `LambdaPi1` limiter is a middle-ground option between the other two.
///    It does not try to avoid limiting as much as `LambdaPiN`, but it allows
///    larger slopes in the data than `Muscl`.
///
/// For all three types of minmod limiter the "total variation bound in the
/// means" (TVBM) correction is implemented, enabling the limiter to avoid
/// limiting away smooth extrema in the solution that would otherwise look like
/// spurious oscillations. The limiter will not reduce the slope (but will still
/// linearize) on elements where the slope is less than \f$m h^2\f$, where
/// \f$m\f$ is the TVBM constant and \f$h\f$ is the size of the DG element.
///
/// The limiter acts in the `Frame::Logical` coordinates, because in these
/// coordinates it is straightforward to formulate the algorithm. This means the
/// limiter can operate on generic deformed grids. However, if the grid is too
/// strongly deformed, some things can start to break down:
/// 1. When an element is deformed so that the Jacobian (from `Frame::Logical`
///    to `Frame::Inertial`) varies across the element, then the limiter fails
///    to be conservative. In other words, the integral of a tensor `u` over the
///    element will change after the limiter activates on `u`. This error is
///    typically small.
/// 2. When there is a sudden change in the size of the elements (perhaps at an
///    h-refinement boundary, or at the boundary between two blocks with very
///    different mappings), a smooth solution in `Frame::Inertial` can appear
///    to have a kink in `Frame::Logical`. The Minmod implementation includes
///    some (untested) tweaks that try to reduce spurious limiter activations
///    near these fake kinks.
///
/// The limiter as implemented expects an element to have one neighbor in each
/// direction, and therefore does not support h-refinement.
///
/// \tparam VolumeDim The number of spatial dimensions.
/// \tparam Tags A typelist of tags specifying the tensors to limit.
///
/// \anchor cockburn_ref [1] B. Cockburn,
/// Discontinuous Galerkin Methods for Convection-Dominated Problems,
/// [Springer (1999)](https://doi.org/10.1007/978-3-662-03882-6_2)
template <size_t VolumeDim, typename... Tags>
class Minmod<VolumeDim, tmpl::list<Tags...>> {
 public:
  struct Type {
    using type = MinmodType;
    static constexpr OptionString help = {"Type of minmod"};
  };
  struct TvbmConstant {
    using type = double;
    static type default_value() { return 0.0; }
    static type lower_bound() { return 0.0; }
    static constexpr OptionString help = {"TVBM constant 'm'"};
  };
  using options = tmpl::list<Type, TvbmConstant>;
  static constexpr OptionString help = {
      "A minmod-based slope limiter.\n"
      "The different types of minmod are more or less aggressive in trying\n"
      "to reduce slopes. The TVBM correction allows the limiter to ignore\n"
      "'small' slopes, and helps to avoid limiting of smooth extrema in the\n"
      "solution.\n"};

  /// \brief Constuct a Minmod slope limiter
  ///
  /// \param minmod_type The type of Minmod slope limiter.
  /// \param tvbm_constant The value of the TVBM constant (default: 0).
  explicit Minmod(const MinmodType minmod_type,
                  const double tvbm_constant = 0.0) noexcept
      : minmod_type_(minmod_type), tvbm_constant_(tvbm_constant) {
    ASSERT(tvbm_constant >= 0.0, "The TVBM constant must be non-negative.");
  }

  Minmod() noexcept = default;
  Minmod(const Minmod& /*rhs*/) = default;
  Minmod& operator=(const Minmod& /*rhs*/) = default;
  Minmod(Minmod&& /*rhs*/) noexcept = default;
  Minmod& operator=(Minmod&& /*rhs*/) noexcept = default;
  ~Minmod() = default;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | minmod_type_;
    p | tvbm_constant_;
  }

  const MinmodType& minmod_type() const noexcept { return minmod_type_; }
  const double& tvbm_constant() const noexcept { return tvbm_constant_; }

  /// \brief Computes data that must be communicated to neighbor elements.
  ///
  /// The minmod limiter needs only the cell-averaged means of the tensors in
  /// each neighboring DG element. This function computes and packages the cell-
  /// averaged data.
  ///
  /// \param means The cell-averaged means of each tensor.
  /// \param tensors The tensors to be averaged and packaged.
  /// \param mesh The mesh on which the tensor values are measured.
  void data_for_neighbors(
      const gsl::not_null<std::add_pointer_t<
          Minmod_detail::tensor_double_from<db::item_type<Tags>>>>... means,
      const db::item_type<Tags>&... tensors, const Mesh<VolumeDim>& mesh) const
      noexcept {
    const auto wrap_compute_mean = [&mesh](const auto& mean,
                                           const auto& tensor) noexcept {
      for (size_t i = 0; i < tensor.size(); ++i) {
        (*mean)[i] = mean_value(tensor[i], mesh);
      }
      return '0';
    };
    expand_pack(wrap_compute_mean(means, tensors)...);
  }

  /// \brief Limits the solution on the element.
  ///
  /// For each component of each tensor, the limiter will (in general) linearize
  /// the data, then possibly reduce its slope, dimension-by-dimension, until it
  /// no longer looks oscillatory.
  ///
  /// \param tensors The tensors to be limited.
  /// \param neighbor_tensors The tensor cell-averages from each neighbor.
  /// \param element The element on which the tensors to limit live.
  /// \param mesh The mesh on which the tensor values are measured.
  /// \param logical_coords The logical coordinates of the mesh gridpoints.
  /// \param element_size The size of the element, in the inertial coordinates.
  /// \param neighbor_sizes The sizes of the neighboring elements.
  ///
  /// \return whether the limiter modified the solution or not.
  ///
  /// \note The return value is false if the limiter knows it has not modified
  /// the solution. True return values can indicate:
  /// - The solution was limited to reduce the slope, whether by a large factor
  ///   or by a factor only roundoff away from unity.
  /// - The solution was linearized but not limited.
  /// - The solution is identical to the input, if the input was a linear
  ///   function on a higher-order mesh, so that the limiter cannot know that
  ///   the linearization step did not actually modify the data. This is
  ///   somewhat contrived and is unlikely to occur outside of code tests or
  ///   test cases with very clean initial data.
  bool apply(
      const gsl::not_null<std::add_pointer_t<db::item_type<Tags>>>... tensors,
      const std::unordered_map<Direction<VolumeDim>,
                               Minmod_detail::tensor_double_from<
                                   db::item_type<Tags>>>&... neighbor_tensors,
      const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
      const tnsr::I<DataVector, VolumeDim, Frame::Logical>& logical_coords,
      const tnsr::I<double, VolumeDim>& element_size,
      const std::unordered_map<Direction<VolumeDim>,
                               tnsr::I<double, VolumeDim>>& neighbor_sizes)
      const noexcept {
    bool limiter_activated = false;
    const auto wrap_limit_one_tensor = [
      this, &element, &mesh, &logical_coords, &element_size, &neighbor_sizes,
      &limiter_activated
    ](const auto& tensor, const auto& neighbor_tensor) noexcept {
      // Get iterators into the local and neighbor tensors, because these are
      // independent from the structure of the tensor being limited.
      const auto tensor_begin = make_not_null(tensor->begin());
      const auto tensor_end = make_not_null(tensor->end());
      const auto neighbor_tensor_begin = [&neighbor_tensor]() noexcept {
        std::unordered_map<Direction<VolumeDim>, gsl::not_null<const double*>>
            result;
        for (const auto& dir_and_tensor : neighbor_tensor) {
          result.insert(
              std::make_pair(dir_and_tensor.first,
                             make_not_null(dir_and_tensor.second.cbegin())));
        }
        return result;
      }
      ();

      limiter_activated = Minmod_detail::limit_one_tensor<VolumeDim>(
                              tensor_begin, tensor_end, neighbor_tensor_begin,
                              minmod_type_, tvbm_constant_, element, mesh,
                              logical_coords, element_size, neighbor_sizes) or
                          limiter_activated;
      return '0';
    };
    expand_pack(wrap_limit_one_tensor(tensors, neighbor_tensors)...);
    return limiter_activated;
  }

 private:
  MinmodType minmod_type_;
  double tvbm_constant_;
};

template <size_t VolumeDim, typename TagList>
SPECTRE_ALWAYS_INLINE bool operator==(
    const Minmod<VolumeDim, TagList>& lhs,
    const Minmod<VolumeDim, TagList>& rhs) noexcept {
  return lhs.minmod_type() == rhs.minmod_type() and
         lhs.tvbm_constant() == rhs.tvbm_constant();
}

template <size_t VolumeDim, typename TagList>
SPECTRE_ALWAYS_INLINE bool operator!=(
    const Minmod<VolumeDim, TagList>& lhs,
    const Minmod<VolumeDim, TagList>& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace SlopeLimiters

template <>
struct create_from_yaml<SlopeLimiters::MinmodType> {
  static SlopeLimiters::MinmodType create(const Option& options);
};
