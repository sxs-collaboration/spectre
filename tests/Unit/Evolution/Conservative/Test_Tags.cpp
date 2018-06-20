// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Domain/FaceNormal.hpp"
#include "Evolution/Conservative/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Var1 : db::SimpleTag {
  static std::string name() noexcept { return "Var1"; }
  using type = Scalar<DataVector>;
};

template <size_t Dim, typename Frame>
struct Var2 : db::SimpleTag {
  static std::string name() noexcept { return "Var2"; }
  using type = tnsr::i<DataVector, Dim, Frame>;
};

template <size_t Dim, typename Frame>
using variables_tag = Tags::Variables<tmpl::list<Var1, Var2<Dim, Frame>>>;

template <size_t Dim, typename Frame>
using flux1 = Tags::Flux<Var1, tmpl::size_t<Dim>, Frame>;
template <size_t Dim, typename Frame>
using flux2 = Tags::Flux<Var2<Dim, Frame>, tmpl::size_t<Dim>, Frame>;

template <size_t Dim, typename Frame>
using flux_tag = db::add_tag_prefix<Tags::Flux, variables_tag<Dim, Frame>,
                                    tmpl::size_t<Dim>, Frame>;

// Copy the values in the components of Tensor<double, ...> `in` into
// the `data_index` entry of DataVectors in the Tensor<DataVector,
// ...> `out`, mapping the component (in0, in1, ...) to (in0, in1,
// ..., `extra_indices`).
template <typename OutTensor, typename InTensor>
void copy_into(const gsl::not_null<OutTensor*> out, const InTensor& in,
               const std::array<size_t, OutTensor::rank() - InTensor::rank()>&
                   extra_indices,
               const size_t data_index) noexcept {
  for (auto it = in.begin(); it != in.end(); ++it) {
    const auto in_index = in.get_tensor_index(it);
    std::array<size_t, OutTensor::rank()> out_index{};
    for (size_t i = 0; i < InTensor::rank(); ++i) {
      gsl::at(out_index, i) = gsl::at(in_index, i);
    }
    for (size_t i = 0; i < OutTensor::rank() - InTensor::rank(); ++i) {
      gsl::at(out_index, i + InTensor::rank()) = gsl::at(extra_indices, i);
    }
    out->get(out_index)[data_index] = in.get(in_index);
  }
}

template <size_t Dim, typename Frame>
tnsr::i<double, Dim, Frame> generate_normal(const size_t seed) noexcept {
  tnsr::i<double, Dim, Frame> result{};
  std::iota(result.begin(), result.end(), seed + 2.);
  return result;
}

template <size_t Dim, typename Frame>
tnsr::I<double, Dim, Frame> generate_flux(const size_t seed) noexcept {
  tnsr::I<double, Dim, Frame> result{};
  std::iota(result.begin(), result.end(), seed + 3.);
  return result;
}

template <size_t Dim>
Scalar<double> generate_f_dot_n(const size_t normal_seed,
                                const size_t flux_seed) noexcept {
  double magnitude_normal = 0.;
  double unnormalized_f_dot_n = 0.;
  for (size_t i = 0; i < Dim; ++i) {
    magnitude_normal += square(normal_seed + i + 2);
    unnormalized_f_dot_n += (normal_seed + i + 2) * (flux_seed + i + 3);
  }
  magnitude_normal = sqrt(magnitude_normal);

  return Scalar<double>(unnormalized_f_dot_n / magnitude_normal);
}

template <size_t Dim, typename Frame>
void check() {
  constexpr size_t num_points = 5;
  tnsr::i<DataVector, Dim, Frame> normal(num_points);
  db::item_type<flux_tag<Dim, Frame>> fluxes(num_points);
  Var1::type expected1(num_points);
  typename Var2<Dim, Frame>::type expected2(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    copy_into(make_not_null(&normal), generate_normal<Dim, Frame>(i), {}, i);
    copy_into(make_not_null(&get<flux1<Dim, Frame>>(fluxes)),
              generate_flux<Dim, Frame>(i), {}, i);
    copy_into(make_not_null(&expected1), generate_f_dot_n<Dim>(i, i), {}, i);
    for (size_t j = 0; j < Dim; ++j) {
      copy_into(make_not_null(&get<flux2<Dim, Frame>>(fluxes)),
                generate_flux<Dim, Frame>(i + 10 * j), {{j}}, i);
      copy_into(make_not_null(&expected2), generate_f_dot_n<Dim>(i, i + 10 * j),
                {{j}}, i);
    }
  }

  // Doing this through a DataBox would require a full element to be
  // set up.
  using magnitude_normal_tag =
      Tags::EuclideanMagnitude<Tags::UnnormalizedFaceNormal<Dim, Frame>>;
  const auto magnitude_normal = magnitude_normal_tag::function(normal);
  using normalized_normal_tag =
      Tags::Normalized<Tags::UnnormalizedFaceNormal<Dim, Frame>>;
  const auto normalized_normal =
      normalized_normal_tag::function(normal, magnitude_normal);
  using compute_n_dot_f =
      Tags::ComputeNormalDotFlux<variables_tag<Dim, Frame>, Dim, Frame>;
  static_assert(
      cpp17::is_same_v<typename compute_n_dot_f::argument_tags,
                       tmpl::list<flux_tag<Dim, Frame>, normalized_normal_tag>>,
      "Wrong argument tags");
  const auto result = compute_n_dot_f::function(fluxes, normalized_normal);

  static_assert(
      cpp17::is_base_of_v<
          db::add_tag_prefix<Tags::NormalDotFlux, variables_tag<Dim, Frame>>,
          compute_n_dot_f>,
      "Wrong inheritance");
  CHECK_ITERABLE_APPROX(get<Tags::NormalDotFlux<Var1>>(result), expected1);
  CHECK_ITERABLE_APPROX((get<Tags::NormalDotFlux<Var2<Dim, Frame>>>(result)),
                        expected2);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.ComputeNormalDotFlux", "[Unit][Evolution]") {
  check<1, Frame::Inertial>();
  check<1, Frame::Grid>();
  check<2, Frame::Inertial>();
  check<2, Frame::Grid>();
  check<3, Frame::Inertial>();
  check<3, Frame::Grid>();
}
