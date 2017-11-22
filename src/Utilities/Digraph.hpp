// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

namespace brigand {

template <typename Source, typename Destination, typename Weight = int32_t<1>>
struct edge {
  using source = Source;
  using destination = Destination;
  using weight = Weight;
};

template <int Source, int Destination, int Weight = 1>
using edge_ = edge<int32_t<Source>, int32_t<Destination>, int32_t<Weight>>;

template <typename T>
struct get_source;
template <typename Source, typename Destination, typename Weight>
struct get_source<edge<Source, Destination, Weight>> {
  using type = Source;
};

template <typename T>
struct get_destination;
template <typename Source, typename Destination, typename Weight>
struct get_destination<edge<Source, Destination, Weight>> {
  using type = Destination;
};

template <typename T>
struct get_weight;
template <typename Source, typename Destination, typename Weight>
struct get_weight<edge<Source, Destination, Weight>> {
  using type = Weight;
};

template <typename State, typename Element>
struct add_unique_vertex;

template <typename State, typename Source, typename Destination,
          typename Weight>
struct add_unique_vertex<State, edge<Source, Destination, Weight>> {
  using source = typename if_<found<State, std::is_same<pin<Source>, _1>>,
                              list<>, list<Source>>::type;
  using destination =
      typename if_<found<State, std::is_same<pin<Destination>, _1>>, list<>,
                   list<Destination>>::type;
  using type = append<State, source, destination>;
};

template <class E, class S, class = std::nullptr_t>
struct has_source : std::false_type {};
template <class E, class S>
struct has_source<E, S, Requires<std::is_same<typename E::source, S>::value>>
    : std::true_type {};

template <class E, class D, class = void>
struct has_destination : std::false_type {};
template <class E, class D>
struct has_destination<
    E, D, Requires<std::is_same<typename E::destination, D>::value>>
    : std::true_type {};

template <class E, class S, class D>
struct has_source_and_destination : std::false_type {};
template <template <class...> class E, class S, class D, class W>
struct has_source_and_destination<E<S, D, W>, S, D> : std::true_type {};

template <class edgeList>
struct digraph;

namespace detail {
template <class Graph, class S, class D>
struct get_edge_impl;
template <class S, class D, class edgeList>
struct get_edge_impl<digraph<edgeList>, S, D> {
  using type = find<edgeList, has_source_and_destination<_1, pin<S>, pin<D>>>;
};
}  // namespace detail

namespace detail {
template <class Graph, class S, class D>
struct has_edge_impl;
template <class S, class D, class edgeList>
struct has_edge_impl<digraph<edgeList>, S, D>
    : found<edgeList, has_source_and_destination<_1, pin<S>, pin<D>>> {};
}  // namespace detail

template <class Graph>
struct outgoing_edges_impl;
template <class edgeList>
struct outgoing_edges_impl<digraph<edgeList>> {
  using type = typename digraph<edgeList>::adjacency_list;
};

template <class Graph>
struct ingoing_edges_impl;
template <class edgeList>
struct ingoing_edges_impl<digraph<edgeList>> {
  using type = typename digraph<edgeList>::ingoing_list;
};

// This is what we would like to do, but Intel cannot handle it...
template <class T, template <class...> class F, class... Es>
struct compute_adjacency_list;
template <template <class...> class VertexSeq, class... Vertices, class... Es,
          template <class...> class F>
struct compute_adjacency_list<VertexSeq<Vertices...>, F, Es...> {
  using type = brigand::list<
      brigand::filter<brigand::list<Es...>, F<brigand::_1, pin<Vertices>>>...>;
};

template <template <class...> class List, class... edges>
struct digraph<List<edges...>> {
 public:
  using edge_list = list<edges...>;
  static_assert(is_set<edge_list>::value,
                "Cannot have repeated edges in a digraph");
  using unique_vertex_list =
      fold<edge_list, list<>, add_unique_vertex<_state, _element>>;
  using vertex_count = size<unique_vertex_list>;
  using edge_count = uint32_t<sizeof...(edges)>;

  using adjacency_list =
      compute_adjacency_list<unique_vertex_list, has_source, edges...>;

  using ingoing_list =
      compute_adjacency_list<unique_vertex_list, has_destination, edges...>;

  template <class E>
  static digraph<::brigand::remove<
      list<edges...>,
      detail::get_edge_impl<digraph<List<edges...>>, typename E::source,
                            typename E::destination>>>
      erase(brigand::type_<E>);

  template <class E>
  static digraph<push_back<
      remove<list<edges...>,
             detail::get_edge_impl<digraph<List<edges...>>, typename E::source,
                                   typename E::destination>>,
      E>>
      insert(::brigand::type_<E>);
};

namespace lazy {
template <class Graph, class S, class D>
using has_edge = ::brigand::detail::has_edge_impl<Graph, S, D>;
}  // namespace lazy

/*!
 * \ingroup UtilitiesGroup
 * \brief Check if a digraph has an edge with source `S` and destination `D`
 *
 *
 */
template <class Graph, class S, class D>
using has_edge = typename ::brigand::lazy::has_edge<Graph, S, D>::type;

namespace lazy {
template <class Graph, class S, class D>
using get_edge = ::brigand::detail::get_edge_impl<Graph, S, D>;
}  // namespace lazy

template <class Graph, class S, class D>
using get_edge = typename ::brigand::lazy::get_edge<Graph, S, D>;

template <class Graph, class Vertex>
using outgoing_edges = at<typename Graph::adjacency_list::type,
                          index_of<typename Graph::unique_vertex_list, Vertex>>;

template <class Graph, class Vertex>
using ingoing_edges = at<typename Graph::ingoing_list::type,
                         index_of<typename Graph::unique_vertex_list, Vertex>>;

template <class Graph>
using vertex_count = size<typename Graph::unique_vertex_list>;

template <class Graph>
using edge_count = size<typename Graph::edge_list>;
}  // namespace brigand
