// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <map>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/UpdateAmrDecision.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DirectionMap.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/SegmentId.hpp"
#include "Domain/Side.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep

namespace {

template <size_t VolumeDim>
void check_amr_decision_is_unchanged(
    std::array<amr::Flag, VolumeDim> my_initial_amr_flags,
    const Element<VolumeDim>& element, const ElementId<VolumeDim>& neighbor_id,
    const std::array<amr::Flag, VolumeDim>& neighbor_amr_flags) noexcept {
  const auto expected_updated_flags = my_initial_amr_flags;
  std::stringstream os;
  os << neighbor_amr_flags;
  INFO(os.str());
  CHECK_FALSE(amr::update_amr_decision(make_not_null(&my_initial_amr_flags),
                                       element, neighbor_id,
                                       neighbor_amr_flags));
  CHECK(expected_updated_flags == my_initial_amr_flags);
}

template <size_t VolumeDim>
void check_amr_decision_is_changed(
    std::array<amr::Flag, VolumeDim> my_initial_amr_flags,
    const Element<VolumeDim>& element, const ElementId<VolumeDim>& neighbor_id,
    const std::array<amr::Flag, VolumeDim>& neighbor_amr_flags,
    const std::array<amr::Flag, VolumeDim>& expected_updated_flags) noexcept {
  std::stringstream os;
  os << my_initial_amr_flags << " " << neighbor_amr_flags;
  INFO(os.str());
  CHECK(amr::update_amr_decision(make_not_null(&my_initial_amr_flags), element,
                                 neighbor_id, neighbor_amr_flags));
  CHECK(expected_updated_flags == my_initial_amr_flags);
}

template <size_t VolumeDim>
using changed_flags_t = std::map<std::pair<std::array<amr::Flag, VolumeDim>,
                                           std::array<amr::Flag, VolumeDim>>,
                                 std::array<amr::Flag, VolumeDim>>;
template <size_t VolumeDim>
void check_update_amr_decision(
    const Element<VolumeDim>& element, const ElementId<VolumeDim>& neighbor_id,
    const std::vector<std::array<amr::Flag, VolumeDim>>& all_flags,
    const changed_flags_t<VolumeDim>& changed_flags) noexcept {
  for (const auto& my_flags : all_flags) {
    for (const auto& neighbor_flags : all_flags) {
      auto search =
          changed_flags.find(std::make_pair(my_flags, neighbor_flags));
      if (search == changed_flags.end()) {
        check_amr_decision_is_unchanged(my_flags, element, neighbor_id,
                                        neighbor_flags);
      } else {
        check_amr_decision_is_changed(my_flags, element, neighbor_id,
                                      neighbor_flags, search->second);
      }
    }
  }
}

Element<1> make_element(
    const ElementId<1>& element_id,
    const std::unordered_set<ElementId<1>>& lower_xi_neighbor_ids,
    const std::unordered_set<ElementId<1>>& upper_xi_neighbor_ids) noexcept {
  DirectionMap<1, Neighbors<1>> neighbors;
  if (not lower_xi_neighbor_ids.empty()) {
    neighbors.emplace(Direction<1>::lower_xi(),
                      Neighbors<1>{{lower_xi_neighbor_ids}, {}});
  }
  if (not upper_xi_neighbor_ids.empty()) {
    neighbors.emplace(Direction<1>::upper_xi(),
                      Neighbors<1>{{upper_xi_neighbor_ids}, {}});
  }
  return Element<1>{element_id, std::move(neighbors)};
}

Element<2> make_element(
    const ElementId<2>& element_id,
    const std::unordered_set<ElementId<2>>& lower_xi_neighbor_ids,
    const std::unordered_set<ElementId<2>>& upper_xi_neighbor_ids,
    const std::unordered_set<ElementId<2>>& lower_eta_neighbor_ids,
    const std::unordered_set<ElementId<2>>& upper_eta_neighbor_ids) noexcept {
  DirectionMap<2, Neighbors<2>> neighbors;
  if (not lower_xi_neighbor_ids.empty()) {
    neighbors.emplace(Direction<2>::lower_xi(),
                      Neighbors<2>{{lower_xi_neighbor_ids}, {}});
  }
  if (not upper_xi_neighbor_ids.empty()) {
    neighbors.emplace(Direction<2>::upper_xi(),
                      Neighbors<2>{{upper_xi_neighbor_ids}, {}});
  }
  if (not lower_eta_neighbor_ids.empty()) {
    neighbors.emplace(Direction<2>::lower_eta(),
                      Neighbors<2>{{lower_eta_neighbor_ids}, {}});
  }
  if (not upper_eta_neighbor_ids.empty()) {
    neighbors.emplace(Direction<2>::upper_eta(),
                      Neighbors<2>{{upper_eta_neighbor_ids}, {}});
  }
  return Element<2>{element_id, std::move(neighbors)};
}

void test_update_amr_decision_1d() noexcept {
  const std::array<amr::Flag, 1> split{{amr::Flag::Split}};
  const std::array<amr::Flag, 1> join{{amr::Flag::Join}};
  const std::array<amr::Flag, 1> stay{{amr::Flag::DoNothing}};
  const std::vector<std::array<amr::Flag, 1>> all_flags{split, join, stay};

  const SegmentId x_segment{3, 5};
  const SegmentId x_cousin{3, 6};

  const ElementId<1> element_id{0, {{x_segment}}};
  const ElementId<1> id_lx_s{0, {{x_segment.id_of_sibling()}}};
  const ElementId<1> id_ux_c{0, {{x_cousin}}};

  auto element = make_element(element_id, {id_lx_s}, {id_ux_c});
  // changed flags: first flags of pair is initial_amr_flags
  //                second flags of pair is neighbor_amr_flags
  //                last flags are the new amr_flags for the element
  check_update_amr_decision(
      element, id_lx_s, all_flags,
      changed_flags_t<1>{{{join, stay}, stay}, {{join, split}, stay}});
  check_update_amr_decision(element, id_ux_c, all_flags,
                            changed_flags_t<1>{{{join, split}, stay}});

  const ElementId<1> id_lx_n{0, {{x_segment.id_of_abutting_nibling()}}};
  const ElementId<1> id_ux_cp{0, {{x_cousin.id_of_parent()}}};
  element = make_element(element_id, {id_lx_n}, {id_ux_cp});
  check_update_amr_decision(element, id_lx_n, all_flags,
                            changed_flags_t<1>{{{join, join}, stay},
                                               {{join, stay}, stay},
                                               {{join, split}, split},
                                               {{stay, split}, split}});
  check_update_amr_decision(element, id_ux_cp, all_flags, changed_flags_t<1>{});

  // note having no lower neighbor is okay for this test
  const ElementId<1> id_ux_cc{0, {{x_cousin.id_of_child(Side::Lower)}}};
  element = make_element(element_id, {}, {id_ux_cc});
  check_update_amr_decision(element, id_ux_cc, all_flags,
                            changed_flags_t<1>{{{join, stay}, stay},
                                               {{join, split}, split},
                                               {{stay, split}, split}});
}

void test_update_amr_decision_2d() noexcept {
  const std::array<amr::Flag, 2> split_split{
      {amr::Flag::Split, amr::Flag::Split}};
  const std::array<amr::Flag, 2> join_split{
      {amr::Flag::Join, amr::Flag::Split}};
  const std::array<amr::Flag, 2> stay_split{
      {amr::Flag::DoNothing, amr::Flag::Split}};
  const std::array<amr::Flag, 2> split_stay{
      {amr::Flag::Split, amr::Flag::DoNothing}};
  const std::array<amr::Flag, 2> join_stay{
      {amr::Flag::Join, amr::Flag::DoNothing}};
  const std::array<amr::Flag, 2> stay_stay{
      {amr::Flag::DoNothing, amr::Flag::DoNothing}};
  const std::array<amr::Flag, 2> split_join{
      {amr::Flag::Split, amr::Flag::Join}};
  const std::array<amr::Flag, 2> join_join{{amr::Flag::Join, amr::Flag::Join}};
  const std::array<amr::Flag, 2> stay_join{
      {amr::Flag::DoNothing, amr::Flag::Join}};

  const std::vector<std::array<amr::Flag, 2>> all_flags{
      split_split, join_split, stay_split, split_stay, join_stay,
      stay_stay,   split_join, join_join,  stay_join};

  const SegmentId x_segment{3, 5};
  const SegmentId x_cousin{3, 6};
  const SegmentId y_segment{6, 15};
  const SegmentId y_cousin{6, 16};

  const ElementId<2> element_id{0, {{x_segment, y_segment}}};

  const ElementId<2> id_lx_s_s{0, {{x_segment.id_of_sibling(), y_segment}}};
  const ElementId<2> id_ux_c_s{0, {{x_cousin, y_segment}}};
  const ElementId<2> id_ly_s_n{
      0, {{x_segment, y_segment.id_of_abutting_nibling()}}};
  const ElementId<2> id_uy_s_cp{0, {{x_segment, y_cousin.id_of_parent()}}};
  auto element = make_element(element_id, {id_lx_s_s}, {id_ux_c_s}, {id_ly_s_n},
                              {id_uy_s_cp});

  // changed flags: first flags of pair are initial_amr_flags
  //                second flags of pair are neighbor_amr_flags
  //                last flags are the new amr_flags for the element

  // neighbor same refinement in x and y, sibling side in x
  check_update_amr_decision<2>(
      element, id_lx_s_s, all_flags,
      changed_flags_t<2>{{{join_split, split_split}, stay_split},
                         {{join_split, stay_split}, stay_split},
                         {{join_split, split_stay}, stay_split},
                         {{join_split, join_stay}, stay_split},
                         {{join_split, stay_stay}, stay_split},
                         {{join_split, split_join}, stay_split},
                         {{join_split, join_join}, stay_split},
                         {{join_split, stay_join}, stay_split},
                         {{join_stay, split_split}, stay_stay},
                         {{join_stay, stay_split}, stay_stay},
                         {{join_stay, join_split}, stay_stay},
                         {{join_stay, split_stay}, stay_stay},
                         {{join_stay, stay_stay}, stay_stay},
                         {{join_stay, split_join}, stay_stay},
                         {{join_stay, join_join}, stay_stay},
                         {{join_stay, stay_join}, stay_stay},
                         {{split_join, split_split}, split_stay},
                         {{split_join, join_split}, split_stay},
                         {{split_join, stay_split}, split_stay},
                         {{join_join, split_split}, stay_stay},
                         {{join_join, join_split}, stay_stay},
                         {{join_join, stay_split}, stay_stay},
                         {{join_join, split_stay}, stay_join},
                         {{join_join, join_stay}, stay_join},
                         {{join_join, stay_stay}, stay_join},
                         {{join_join, split_join}, stay_join},
                         {{join_join, stay_join}, stay_join},
                         {{stay_join, split_split}, stay_stay},
                         {{stay_join, join_split}, stay_stay},
                         {{stay_join, stay_split}, stay_stay}});
  // neighbor same refinement in x and y, non-sibling side in x
  check_update_amr_decision<2>(
      element, id_ux_c_s, all_flags,
      changed_flags_t<2>{{{join_split, split_split}, stay_split},
                         {{join_split, split_stay}, stay_split},
                         {{join_split, split_join}, stay_split},
                         {{join_stay, split_split}, stay_stay},
                         {{join_stay, split_stay}, stay_stay},
                         {{join_stay, split_join}, stay_stay},
                         {{split_join, split_split}, split_stay},
                         {{split_join, join_split}, split_stay},
                         {{split_join, stay_split}, split_stay},
                         {{join_join, split_split}, stay_stay},
                         {{join_join, join_split}, join_stay},
                         {{join_join, stay_split}, join_stay},
                         {{join_join, split_stay}, stay_join},
                         {{join_join, split_join}, stay_join},
                         {{stay_join, split_split}, stay_stay},
                         {{stay_join, join_split}, stay_stay},
                         {{stay_join, stay_split}, stay_stay}});
  // neighbor same in x, coarser in y, sibling side of y
  check_update_amr_decision<2>(
      element, id_ly_s_n, all_flags,
      changed_flags_t<2>{{{join_split, split_split}, stay_split},
                         {{join_split, split_stay}, stay_split},
                         {{join_split, split_join}, stay_split},
                         {{split_stay, split_split}, split_split},
                         {{split_stay, join_split}, split_split},
                         {{split_stay, stay_split}, split_split},
                         {{join_stay, split_split}, stay_split},
                         {{join_stay, stay_split}, join_split},
                         {{join_stay, join_split}, join_split},
                         {{join_stay, split_stay}, stay_stay},
                         {{join_stay, split_join}, stay_stay},
                         {{stay_stay, split_split}, stay_split},
                         {{stay_stay, join_split}, stay_split},
                         {{stay_stay, stay_split}, stay_split},
                         {{split_join, split_split}, split_split},
                         {{split_join, join_split}, split_split},
                         {{split_join, stay_split}, split_split},
                         {{split_join, split_stay}, split_stay},
                         {{split_join, join_stay}, split_stay},
                         {{split_join, stay_stay}, split_stay},
                         {{split_join, split_join}, split_stay},
                         {{split_join, join_join}, split_stay},
                         {{split_join, stay_join}, split_stay},
                         {{join_join, split_split}, stay_split},
                         {{join_join, join_split}, join_split},
                         {{join_join, stay_split}, join_split},
                         {{join_join, split_stay}, stay_stay},
                         {{join_join, join_stay}, join_stay},
                         {{join_join, stay_stay}, join_stay},
                         {{join_join, split_join}, stay_stay},
                         {{join_join, join_join}, join_stay},
                         {{join_join, stay_join}, join_stay},
                         {{stay_join, split_split}, stay_split},
                         {{stay_join, join_split}, stay_split},
                         {{stay_join, stay_split}, stay_split},
                         {{stay_join, split_stay}, stay_stay},
                         {{stay_join, join_stay}, stay_stay},
                         {{stay_join, stay_stay}, stay_stay},
                         {{stay_join, split_join}, stay_stay},
                         {{stay_join, join_join}, stay_stay},
                         {{stay_join, stay_join}, stay_stay}});
  // neighbor same in x, coarser in y, non-sibling side of y
  check_update_amr_decision<2>(
      element, id_uy_s_cp, all_flags,
      changed_flags_t<2>{{{join_split, split_split}, stay_split},
                         {{join_split, split_stay}, stay_split},
                         {{join_split, split_join}, stay_split},
                         {{join_stay, split_split}, stay_stay},
                         {{join_stay, split_stay}, stay_stay},
                         {{join_stay, split_join}, stay_stay},
                         {{join_join, split_split}, stay_join},
                         {{join_join, split_stay}, stay_join},
                         {{join_join, split_join}, stay_join}});

  const ElementId<2> id_lx_s_p{
      0, {{x_segment.id_of_sibling(), y_segment.id_of_parent()}}};
  const ElementId<2> id_ly_p_n{
      0, {{x_segment.id_of_parent(), y_segment.id_of_abutting_nibling()}}};
  const ElementId<2> id_ux_c_p{0, {{x_cousin, y_segment.id_of_parent()}}};
  const ElementId<2> id_uy_p_cp{
      0, {{x_segment.id_of_parent(), y_cousin.id_of_parent()}}};
  element = make_element(element_id, {id_lx_s_p}, {id_ux_c_p}, {id_ly_p_n},
                         {id_uy_p_cp});
  // neighbor same in x, coarser in y, sibling side of x
  check_update_amr_decision<2>(
      element, id_lx_s_p, all_flags,
      changed_flags_t<2>{{{join_split, split_split}, stay_split},
                         {{join_split, join_split}, stay_split},
                         {{join_split, stay_split}, stay_split},
                         {{join_split, split_stay}, stay_split},
                         {{join_split, join_stay}, stay_split},
                         {{join_split, stay_stay}, stay_split},
                         {{join_split, split_join}, stay_split},
                         {{join_split, join_join}, stay_split},
                         {{join_split, stay_join}, stay_split},
                         {{join_stay, split_split}, stay_stay},
                         {{join_stay, stay_split}, stay_stay},
                         {{join_stay, split_stay}, stay_stay},
                         {{join_stay, stay_stay}, stay_stay},
                         {{join_stay, join_stay}, stay_stay},
                         {{join_stay, split_join}, stay_stay},
                         {{join_stay, join_join}, stay_stay},
                         {{join_stay, stay_join}, stay_stay},
                         {{join_join, split_split}, stay_join},
                         {{join_join, join_split}, stay_join},
                         {{join_join, stay_split}, stay_join},
                         {{join_join, split_stay}, stay_join},
                         {{join_join, stay_stay}, stay_join},
                         {{join_join, split_join}, stay_join},
                         {{join_join, join_join}, stay_join},
                         {{join_join, stay_join}, stay_join}});
  // neighbor same in x, coarser in y, non-sibling side of x
  check_update_amr_decision<2>(
      element, id_ux_c_p, all_flags,
      changed_flags_t<2>{{{join_split, split_split}, stay_split},
                         {{join_split, split_stay}, stay_split},
                         {{join_split, split_join}, stay_split},
                         {{join_stay, split_split}, stay_stay},
                         {{join_stay, split_stay}, stay_stay},
                         {{join_stay, split_join}, stay_stay},
                         {{join_join, split_split}, stay_join},
                         {{join_join, split_stay}, stay_join},
                         {{join_join, split_join}, stay_join}});
  // neighbor coarser in x, finer in y, sibling side of y
  check_update_amr_decision<2>(
      element, id_ly_p_n, all_flags,
      changed_flags_t<2>{{{split_stay, split_split}, split_split},
                         {{split_stay, join_split}, split_split},
                         {{split_stay, stay_split}, split_split},
                         {{join_stay, split_split}, join_split},
                         {{join_stay, stay_split}, join_split},
                         {{join_stay, join_split}, join_split},
                         {{stay_stay, split_split}, stay_split},
                         {{stay_stay, join_split}, stay_split},
                         {{stay_stay, stay_split}, stay_split},
                         {{split_join, split_split}, split_split},
                         {{split_join, join_split}, split_split},
                         {{split_join, stay_split}, split_split},
                         {{split_join, split_stay}, split_stay},
                         {{split_join, join_stay}, split_stay},
                         {{split_join, stay_stay}, split_stay},
                         {{split_join, split_join}, split_stay},
                         {{split_join, join_join}, split_stay},
                         {{split_join, stay_join}, split_stay},
                         {{join_join, split_split}, join_split},
                         {{join_join, join_split}, join_split},
                         {{join_join, stay_split}, join_split},
                         {{join_join, split_stay}, join_stay},
                         {{join_join, join_stay}, join_stay},
                         {{join_join, stay_stay}, join_stay},
                         {{join_join, split_join}, join_stay},
                         {{join_join, join_join}, join_stay},
                         {{join_join, stay_join}, join_stay},
                         {{stay_join, split_split}, stay_split},
                         {{stay_join, join_split}, stay_split},
                         {{stay_join, stay_split}, stay_split},
                         {{stay_join, split_stay}, stay_stay},
                         {{stay_join, join_stay}, stay_stay},
                         {{stay_join, stay_stay}, stay_stay},
                         {{stay_join, split_join}, stay_stay},
                         {{stay_join, join_join}, stay_stay},
                         {{stay_join, stay_join}, stay_stay}});
  // neighbor coarser in x and y, non-sibling side -f y
  check_update_amr_decision<2>(element, id_uy_p_cp, all_flags,
                               changed_flags_t<2>{});

  const ElementId<2> id_lx_s_cl{
      0, {{x_segment.id_of_sibling(), y_segment.id_of_child(Side::Lower)}}};
  const ElementId<2> id_lx_s_cu{
      0, {{x_segment.id_of_sibling(), y_segment.id_of_child(Side::Upper)}}};
  const ElementId<2> id_ly_cl_n{0,
                                {{x_segment.id_of_child(Side::Lower),
                                  y_segment.id_of_abutting_nibling()}}};
  const ElementId<2> id_ly_cu_n{0,
                                {{x_segment.id_of_child(Side::Upper),
                                  y_segment.id_of_abutting_nibling()}}};
  const ElementId<2> id_ux_c_cl{
      0, {{x_cousin, y_segment.id_of_child(Side::Lower)}}};
  const ElementId<2> id_ux_c_cu{
      0, {{x_cousin, y_segment.id_of_child(Side::Upper)}}};
  const ElementId<2> id_uy_cl_cp{
      0, {{x_segment.id_of_child(Side::Lower), y_cousin.id_of_parent()}}};
  const ElementId<2> id_uy_cu_cp{
      0, {{x_segment.id_of_child(Side::Upper), y_cousin.id_of_parent()}}};
  element = make_element(element_id, {id_lx_s_cl, id_lx_s_cu},
                         {id_ux_c_cl, id_ux_c_cu}, {id_ly_cl_n, id_ly_cu_n},
                         {id_uy_cl_cp, id_uy_cu_cp});
  // neighbor same in x, finer in y, sibling side of x
  check_update_amr_decision<2>(
      element, id_lx_s_cl, all_flags,
      changed_flags_t<2>{{{join_split, split_split}, stay_split},
                         {{join_split, join_split}, stay_split},
                         {{join_split, stay_split}, stay_split},
                         {{join_split, split_stay}, stay_split},
                         {{join_split, stay_stay}, stay_split},
                         {{join_split, split_join}, stay_split},
                         {{join_split, join_join}, stay_split},
                         {{join_split, stay_join}, stay_split},
                         {{split_stay, split_split}, split_split},
                         {{split_stay, join_split}, split_split},
                         {{split_stay, stay_split}, split_split},
                         {{join_stay, split_split}, stay_split},
                         {{join_stay, join_split}, stay_split},
                         {{join_stay, stay_split}, stay_split},
                         {{join_stay, split_stay}, stay_stay},
                         {{join_stay, join_stay}, stay_stay},
                         {{join_stay, stay_stay}, stay_stay},
                         {{join_stay, split_join}, stay_stay},
                         {{join_stay, stay_join}, stay_stay},
                         {{stay_stay, split_split}, stay_split},
                         {{stay_stay, join_split}, stay_split},
                         {{stay_stay, stay_split}, stay_split},
                         {{split_join, split_split}, split_split},
                         {{split_join, join_split}, split_split},
                         {{split_join, stay_split}, split_split},
                         {{split_join, split_stay}, split_stay},
                         {{split_join, join_stay}, split_stay},
                         {{split_join, stay_stay}, split_stay},
                         {{join_join, split_split}, stay_split},
                         {{join_join, join_split}, stay_split},
                         {{join_join, stay_split}, stay_split},
                         {{join_join, split_stay}, stay_stay},
                         {{join_join, join_stay}, stay_stay},
                         {{join_join, stay_stay}, stay_stay},
                         {{join_join, split_join}, stay_join},
                         {{join_join, join_join}, stay_join},
                         {{join_join, stay_join}, stay_join},
                         {{stay_join, split_split}, stay_split},
                         {{stay_join, join_split}, stay_split},
                         {{stay_join, stay_split}, stay_split},
                         {{stay_join, split_stay}, stay_stay},
                         {{stay_join, join_stay}, stay_stay},
                         {{stay_join, stay_stay}, stay_stay}});
  // neighbor same in x, finer in y, non-sibling side of x
  check_update_amr_decision<2>(
      element, id_ux_c_cu, all_flags,
      changed_flags_t<2>{{{join_split, split_split}, stay_split},
                         {{join_split, split_stay}, stay_split},
                         {{join_split, split_join}, stay_split},
                         {{split_stay, split_split}, split_split},
                         {{split_stay, join_split}, split_split},
                         {{split_stay, stay_split}, split_split},
                         {{join_stay, split_split}, stay_split},
                         {{join_stay, join_split}, join_split},
                         {{join_stay, stay_split}, join_split},
                         {{join_stay, split_stay}, stay_stay},
                         {{join_stay, split_join}, stay_stay},
                         {{stay_stay, split_split}, stay_split},
                         {{stay_stay, join_split}, stay_split},
                         {{stay_stay, stay_split}, stay_split},
                         {{split_join, split_split}, split_split},
                         {{split_join, join_split}, split_split},
                         {{split_join, stay_split}, split_split},
                         {{split_join, split_stay}, split_stay},
                         {{split_join, join_stay}, split_stay},
                         {{split_join, stay_stay}, split_stay},
                         {{join_join, split_split}, stay_split},
                         {{join_join, join_split}, join_split},
                         {{join_join, stay_split}, join_split},
                         {{join_join, split_stay}, stay_stay},
                         {{join_join, join_stay}, join_stay},
                         {{join_join, stay_stay}, join_stay},
                         {{join_join, split_join}, stay_join},
                         {{stay_join, split_split}, stay_split},
                         {{stay_join, join_split}, stay_split},
                         {{stay_join, stay_split}, stay_split},
                         {{stay_join, split_stay}, stay_stay},
                         {{stay_join, join_stay}, stay_stay},
                         {{stay_join, stay_stay}, stay_stay}});
  // neighbor finer in x and y, sibling side of y
  check_update_amr_decision<2>(
      element, id_ly_cl_n, all_flags,
      changed_flags_t<2>{{{join_split, split_split}, split_split},
                         {{join_split, stay_split}, stay_split},
                         {{join_split, split_stay}, split_split},
                         {{join_split, stay_stay}, stay_split},
                         {{join_split, split_join}, split_split},
                         {{join_split, stay_join}, stay_split},
                         {{stay_split, split_split}, split_split},
                         {{stay_split, split_stay}, split_split},
                         {{stay_split, split_join}, split_split},
                         {{split_stay, split_split}, split_split},
                         {{split_stay, join_split}, split_split},
                         {{split_stay, stay_split}, split_split},
                         {{join_stay, split_split}, split_split},
                         {{join_stay, join_split}, join_split},
                         {{join_stay, stay_split}, stay_split},
                         {{join_stay, split_stay}, split_stay},
                         {{join_stay, stay_stay}, stay_stay},
                         {{join_stay, split_join}, split_stay},
                         {{join_stay, stay_join}, stay_stay},
                         {{stay_stay, split_split}, split_split},
                         {{stay_stay, join_split}, stay_split},
                         {{stay_stay, stay_split}, stay_split},
                         {{stay_stay, split_stay}, split_stay},
                         {{stay_stay, split_join}, split_stay},
                         {{split_join, split_split}, split_split},
                         {{split_join, join_split}, split_split},
                         {{split_join, stay_split}, split_split},
                         {{split_join, split_stay}, split_stay},
                         {{split_join, join_stay}, split_stay},
                         {{split_join, stay_stay}, split_stay},
                         {{split_join, split_join}, split_stay},
                         {{split_join, join_join}, split_stay},
                         {{split_join, stay_join}, split_stay},
                         {{join_join, split_split}, split_split},
                         {{join_join, join_split}, join_split},
                         {{join_join, stay_split}, stay_split},
                         {{join_join, split_stay}, split_stay},
                         {{join_join, join_stay}, join_stay},
                         {{join_join, stay_stay}, stay_stay},
                         {{join_join, split_join}, split_stay},
                         {{join_join, join_join}, join_stay},
                         {{join_join, stay_join}, stay_stay},
                         {{stay_join, split_split}, split_split},
                         {{stay_join, join_split}, stay_split},
                         {{stay_join, stay_split}, stay_split},
                         {{stay_join, split_stay}, split_stay},
                         {{stay_join, join_stay}, stay_stay},
                         {{stay_join, stay_stay}, stay_stay},
                         {{stay_join, split_join}, split_stay},
                         {{stay_join, join_join}, stay_stay},
                         {{stay_join, stay_join}, stay_stay}});
  // neighbor finer in x, coarser in y, non-sibling side of y
  check_update_amr_decision<2>(
      element, id_uy_cu_cp, all_flags,
      changed_flags_t<2>{{{join_split, split_split}, split_split},
                         {{join_split, stay_split}, stay_split},
                         {{join_split, split_stay}, split_split},
                         {{join_split, stay_stay}, stay_split},
                         {{join_split, split_join}, split_split},
                         {{join_split, stay_join}, stay_split},
                         {{stay_split, split_split}, split_split},
                         {{stay_split, split_stay}, split_split},
                         {{stay_split, split_join}, split_split},
                         {{join_stay, split_split}, split_stay},
                         {{join_stay, stay_split}, stay_stay},
                         {{join_stay, split_stay}, split_stay},
                         {{join_stay, stay_stay}, stay_stay},
                         {{join_stay, split_join}, split_stay},
                         {{join_stay, stay_join}, stay_stay},
                         {{stay_stay, split_split}, split_stay},
                         {{stay_stay, split_stay}, split_stay},
                         {{stay_stay, split_join}, split_stay},
                         {{join_join, split_split}, split_join},
                         {{join_join, stay_split}, stay_join},
                         {{join_join, split_stay}, split_join},
                         {{join_join, stay_stay}, stay_join},
                         {{join_join, split_join}, split_join},
                         {{join_join, stay_join}, stay_join},
                         {{stay_join, split_split}, split_join},
                         {{stay_join, split_stay}, split_join},
                         {{stay_join, split_join}, split_join}});

  const ElementId<2> id_ux_cc_s{
      0, {{x_cousin.id_of_child(Side::Lower), y_segment}}};
  const ElementId<2> id_uy_cl_cc{0,
                                 {{x_segment.id_of_child(Side::Lower),
                                   y_cousin.id_of_child(Side::Lower)}}};
  const ElementId<2> id_uy_cu_cc{0,
                                 {{x_segment.id_of_child(Side::Upper),
                                   y_cousin.id_of_child(Side::Lower)}}};

  element = make_element(element_id, {}, {id_ux_cc_s}, {},
                         {id_uy_cl_cc, id_uy_cu_cc});
  // neighbor finer in x, same in y, non-sibling side of x
  check_update_amr_decision<2>(
      element, id_ux_cc_s, all_flags,
      changed_flags_t<2>{{{join_split, split_split}, split_split},
                         {{join_split, stay_split}, stay_split},
                         {{join_split, split_stay}, split_split},
                         {{join_split, stay_stay}, stay_split},
                         {{join_split, split_join}, split_split},
                         {{join_split, stay_join}, stay_split},
                         {{stay_split, split_split}, split_split},
                         {{stay_split, split_stay}, split_split},
                         {{stay_split, split_join}, split_split},
                         {{join_stay, split_split}, split_stay},
                         {{join_stay, stay_split}, stay_stay},
                         {{join_stay, split_stay}, split_stay},
                         {{join_stay, stay_stay}, stay_stay},
                         {{join_stay, split_join}, split_stay},
                         {{join_stay, stay_join}, stay_stay},
                         {{stay_stay, split_split}, split_stay},
                         {{stay_stay, split_stay}, split_stay},
                         {{stay_stay, split_join}, split_stay},
                         {{split_join, split_split}, split_stay},
                         {{split_join, join_split}, split_stay},
                         {{split_join, stay_split}, split_stay},
                         {{join_join, split_split}, split_stay},
                         {{join_join, join_split}, join_stay},
                         {{join_join, stay_split}, stay_stay},
                         {{join_join, split_stay}, split_join},
                         {{join_join, stay_stay}, stay_join},
                         {{join_join, split_join}, split_join},
                         {{join_join, stay_join}, stay_join},
                         {{stay_join, split_split}, split_stay},
                         {{stay_join, join_split}, stay_stay},
                         {{stay_join, stay_split}, stay_stay},
                         {{stay_join, split_stay}, split_join},
                         {{stay_join, split_join}, split_join}});
  // neighbor finer in x and y, non-sibling side of y
  check_update_amr_decision<2>(
      element, id_uy_cu_cc, all_flags,
      changed_flags_t<2>{{{join_split, split_split}, split_split},
                         {{join_split, stay_split}, stay_split},
                         {{join_split, split_stay}, split_split},
                         {{join_split, stay_stay}, stay_split},
                         {{join_split, split_join}, split_split},
                         {{join_split, stay_join}, stay_split},
                         {{stay_split, split_split}, split_split},
                         {{stay_split, split_stay}, split_split},
                         {{stay_split, split_join}, split_split},
                         {{split_stay, split_split}, split_split},
                         {{split_stay, join_split}, split_split},
                         {{split_stay, stay_split}, split_split},
                         {{join_stay, split_split}, split_split},
                         {{join_stay, join_split}, join_split},
                         {{join_stay, stay_split}, stay_split},
                         {{join_stay, split_stay}, split_stay},
                         {{join_stay, stay_stay}, stay_stay},
                         {{join_stay, split_join}, split_stay},
                         {{join_stay, stay_join}, stay_stay},
                         {{stay_stay, split_split}, split_split},
                         {{stay_stay, join_split}, stay_split},
                         {{stay_stay, stay_split}, stay_split},
                         {{stay_stay, split_stay}, split_stay},
                         {{stay_stay, split_join}, split_stay},
                         {{split_join, split_split}, split_split},
                         {{split_join, join_split}, split_split},
                         {{split_join, stay_split}, split_split},
                         {{split_join, split_stay}, split_stay},
                         {{split_join, join_stay}, split_stay},
                         {{split_join, stay_stay}, split_stay},
                         {{join_join, split_split}, split_split},
                         {{join_join, join_split}, join_split},
                         {{join_join, stay_split}, stay_split},
                         {{join_join, split_stay}, split_stay},
                         {{join_join, join_stay}, join_stay},
                         {{join_join, stay_stay}, stay_stay},
                         {{join_join, split_join}, split_join},
                         {{join_join, stay_join}, stay_join},
                         {{stay_join, split_split}, split_split},
                         {{stay_join, join_split}, stay_split},
                         {{stay_join, stay_split}, stay_split},
                         {{stay_join, split_stay}, split_stay},
                         {{stay_join, join_stay}, stay_stay},
                         {{stay_join, stay_stay}, stay_stay},
                         {{stay_join, split_join}, split_join}});

  const ElementId<2> id_uy_p_cc{
      0, {{x_segment.id_of_parent(), y_cousin.id_of_child(Side::Lower)}}};
  element = make_element(element_id, {}, {}, {}, {id_uy_p_cc});
  // neighbor coarser in x, finer in y, non-sibling side of y
  check_update_amr_decision<2>(
      element, id_uy_p_cc, all_flags,
      changed_flags_t<2>{{{split_stay, split_split}, split_split},
                         {{split_stay, join_split}, split_split},
                         {{split_stay, stay_split}, split_split},
                         {{join_stay, split_split}, join_split},
                         {{join_stay, join_split}, join_split},
                         {{join_stay, stay_split}, join_split},
                         {{stay_stay, split_split}, stay_split},
                         {{stay_stay, join_split}, stay_split},
                         {{stay_stay, stay_split}, stay_split},
                         {{split_join, split_split}, split_split},
                         {{split_join, join_split}, split_split},
                         {{split_join, stay_split}, split_split},
                         {{split_join, split_stay}, split_stay},
                         {{split_join, join_stay}, split_stay},
                         {{split_join, stay_stay}, split_stay},
                         {{join_join, split_split}, join_split},
                         {{join_join, join_split}, join_split},
                         {{join_join, stay_split}, join_split},
                         {{join_join, split_stay}, join_stay},
                         {{join_join, join_stay}, join_stay},
                         {{join_join, stay_stay}, join_stay},
                         {{stay_join, split_split}, stay_split},
                         {{stay_join, join_split}, stay_split},
                         {{stay_join, stay_split}, stay_split},
                         {{stay_join, split_stay}, stay_stay},
                         {{stay_join, join_stay}, stay_stay},
                         {{stay_join, stay_stay}, stay_stay}});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Amr.UpdateAmrDecision", "[Domain][Unit]") {
  test_update_amr_decision_1d();
  test_update_amr_decision_2d();
  // 3d is not tested for every case as the algorithm is independent of
  // dimensions once there is a transverse direction for a neighbor and there
  // are 729 AMR flag combinations for 45 different types of neighbors.
}
