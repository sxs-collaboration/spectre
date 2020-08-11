// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/optional.hpp>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceComputeTags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsCharacteresticSpeeds.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim>
struct CharSpeeds : db::SimpleTag {
  using type = std::array<DataVector, 4>;
};

template <size_t Dim>
struct CharSpeedsCompute : CharSpeeds<Dim>, db::ComputeTag {
  using base = CharSpeeds<Dim>;
  using return_type = std::array<DataVector, 4>;

  static void function(const gsl::not_null<std::array<DataVector, 4>*> result,
                       const tnsr::I<DataVector, Dim, Frame::Inertial>&
                           inertial_coords) noexcept {
    gsl::at(*result, 0) = inertial_coords.get(0);
    for (size_t i = 1; i < 4; ++i) {
      gsl::at(*result, i) = inertial_coords.get(0) + 2.0 * i;
    }
  }
  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>>;
};

template <size_t Dim>
struct Directions : db::SimpleTag {
  static std::string name() noexcept { return "Directions"; }
  using type = std::unordered_set<Direction<Dim>>;
};

template <size_t Dim>
std::unordered_set<Direction<Dim>> get_directions();

template <>
std::unordered_set<Direction<1>> get_directions<1>() {
  return std::unordered_set<Direction<1>>{Direction<1>::upper_xi()};
}

template <>
std::unordered_set<Direction<2>> get_directions<2>() {
  return std::unordered_set<Direction<2>>{Direction<2>::upper_xi(),
                                          Direction<2>::lower_eta()};
}

template <>
std::unordered_set<Direction<3>> get_directions<3>() {
  return std::unordered_set<Direction<3>>{Direction<3>::upper_xi(),
                                          Direction<3>::lower_eta(),
                                          Direction<3>::lower_zeta()};
}

template <size_t Dim, bool MeshIsMoving>
void test_tags() {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);

  TestHelpers::db::test_compute_tag<
      domain::Tags::CharSpeedCompute<CharSpeedsCompute<Dim>, Dim>>(
      "CharSpeeds");

  using simple_tags = db::AddSimpleTags<
      Directions<Dim>,
      domain::Tags::Interface<Directions<Dim>, domain::Tags::MeshVelocity<Dim>>,
      domain::Tags::Interface<Directions<Dim>,
                              domain::Tags::Coordinates<Dim, Frame::Inertial>>,
      domain::Tags::Interface<
          Directions<Dim>,
          Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>>;

  using compute_tags = db::AddComputeTags<domain::Tags::InterfaceCompute<
      Directions<Dim>,
      domain::Tags::CharSpeedCompute<CharSpeedsCompute<Dim>, Dim>>>;

  const DataVector used_for_size(5);

  std::unordered_map<Direction<Dim>,
                     boost::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>>
      mesh_velocity{};
  std::unordered_map<Direction<Dim>, tnsr::I<DataVector, Dim, Frame::Inertial>>
      coordinates{};
  std::unordered_map<Direction<Dim>, tnsr::i<DataVector, Dim, Frame::Inertial>>
      normals{};
  for (const auto& direction : get_directions<Dim>()) {
    if (MeshIsMoving) {
      mesh_velocity[direction] =
          make_with_random_values<tnsr::I<DataVector, Dim, Frame::Inertial>>(
              make_not_null(&generator), make_not_null(&dist), used_for_size);
    } else {
      mesh_velocity[direction] = boost::none;
    }
    coordinates[direction] =
        make_with_random_values<tnsr::I<DataVector, Dim, Frame::Inertial>>(
            make_not_null(&generator), make_not_null(&dist), used_for_size);
    normals[direction] =
        make_with_random_values<tnsr::i<DataVector, Dim, Frame::Inertial>>(
            make_not_null(&generator), make_not_null(&dist), used_for_size);
  }

  const auto box = db::create<simple_tags, compute_tags>(
      get_directions<Dim>(), mesh_velocity, coordinates, normals);

  std::unordered_map<Direction<Dim>, std::array<DataVector, 4>>
      expected_char_speeds{};
  for (const auto& direction : get_directions<Dim>()) {
    CharSpeedsCompute<Dim>::function(
        make_not_null(&expected_char_speeds[direction]),
        coordinates[direction]);
    if (MeshIsMoving) {
      const Scalar<DataVector> normal_dot_velocity =
          dot_product(normals[direction], *(mesh_velocity[direction]));
      for (size_t i = 0; i < expected_char_speeds[direction].size(); ++i) {
        gsl::at(expected_char_speeds[direction], i) -= get(normal_dot_velocity);
      }
    }

    CHECK_ITERABLE_APPROX(
        (db::get<domain::Tags::Interface<Directions<Dim>, CharSpeeds<Dim>>>(box)
             .at(direction)),
        expected_char_speeds.at(direction));
  }
}

SPECTRE_TEST_CASE("Unit.Domain.TagsCharacteresticSpeeds", "[Unit][Actions]") {
  test_tags<1, true>();
  test_tags<2, true>();
  test_tags<3, true>();

  test_tags<1, false>();
  test_tags<2, false>();
  test_tags<3, false>();
}
}  // namespace
