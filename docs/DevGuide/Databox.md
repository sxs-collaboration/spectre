\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Motivation for SpECTRE's DataBox {#databox_foundations}

\tableofcontents

# Introduction {#introduction}
This page walks the user through the iterative process that led to SpECTRE's
DataBox. At each stage, it discusses the advances and challenges that result
from each improvement.

# Towards SpECTRE's DataBox {#towards_spectres_databox}

## Working without DataBoxes {#working_without_databoxes}
In a small C++ program, it is common to use the built-in fundamental types
(bool, int, double, etc.) in computations, and to give variable names to
objects of these types. For example, a section of a small program may look like
this:

\snippet Test_DataBoxDocumentation.cpp working_without_databoxes_small_program_1

What changes as our program's size increases in scale? In SpECTRE, one of our
driving design goals is modularity. In other words, functionality should be
easy to swap in and out as desired. These smaller modules are easier to test,
and keep the code flexible. We could wrap our calculation in such a module,
which would then allow us to decouple the initial setup of the variables from
where they are used in calculations:

\snippet Test_DataBoxDocumentation.cpp working_without_databoxes_mass_compute

\snippet Test_DataBoxDocumentation.cpp working_without_databoxes_accel_compute

Our small program can now be written as:

\snippet Test_DataBoxDocumentation.cpp working_without_databoxes_small_program_2

One advantage is immediate: we are free to add other computation modules that
are independently testable and reusable. As the number of routines grows, we
can even begin to write routines that work on top of existing ones.

\snippet Test_DataBoxDocumentation.cpp working_without_databoxes_force_compute

While we have made progress, two problems arise. Our first problem is that as
the number of quantities grows, it becomes more unwieldy to have to specify
each function argument in our routine. The second problem is worse: the
arguments passed to functions can be transposed and the program will still
compile and run, but produce incorrect output. For example, the following two
lines are equally well-formed from the point of view of the program:

\snippet Test_DataBoxDocumentation.cpp working_without_databoxes_failed_accel

Every time we call `acceleration_compute` we need to make sure we pass in the
arguments in the correct order. In large programs where `acceleration_compute`
is called many times, it becomes inevitable that the arguments will be
accidentally transposed. We can address the two problems described above with a
`std::map`, the first container we'll consider in this series.

##A std::map DataBox {#a_std_map_databox}
We can encapsulate the variables we use in a `std::map`, and the first half of
our small program example now looks like this:

\snippet Test_DataBoxDocumentation.cpp std_map_databox_small_program_1

We have not yet taken full advantage of the encapsulation that `std::map`
provides. We do so by rewriting our other routines to take only a single
argument, i.e. the `std::map` itself:

\snippet Test_DataBoxDocumentation.cpp std_map_databox_mass_compute

\snippet Test_DataBoxDocumentation.cpp std_map_databox_accel_compute

Notice that this solves the problem of having to provide the arguments
in the correct order to every function call of `acceleration_compute`:

\snippet Test_DataBoxDocumentation.cpp std_map_databox_force_compute


Our small program now looks like:

\snippet Test_DataBoxDocumentation.cpp std_map_databox_small_program_2

Within each function, we no longer need to worry about passing in the arguments
in the correct order. This is a great improvement, but our reliance on proper
names does leave us open to the following mistake:

~~~{.c}
// returns 0 without emitting an error!
return naive_databox["MisspelledKey"];
~~~

In the above example, the map is asked to return a value given a key
that does not exist! As written, however, the program is well-formed and no
error is emitted. (In the case of std::map, [a value is created.]
(https://en.cppreference.com/w/cpp/container/map/operator_at)) Because the keys
are indistinguishable from their type alone, the mistake cannot be caught at
compile time. In our example, the mistake won't even be caught at run time. The
run time portion of a SpECTRE calculation will typically be much longer (up to
thousands of times longer!) than the compile time portion, so it is critical to
catch costly mistakes like this as early into the calculation as possible.
Although names encoded as `std::string` cannot be distinguished by the
compiler, names encoded as types *can* be. This is possible with C++'s static
typing, and to take advantage of this we need a container that is
*heterogeneous*, that is, capable of holding objects of different types.

## A std::tuple DataBox {#a_std_tuple_databox}

A well-documented example of a fixed-size heterogeneous container of types is
[std::tuple](https://en.cppreference.com/w/cpp/utility/tuple):

\snippet Test_DataBoxDocumentation.cpp std_tuple_databox_1

The contents of the `std_tuple` are obtained using
[std::get](https://en.cppreference.com/w/cpp/utility/tuple/get):

\snippet Test_DataBoxDocumentation.cpp std_tuple_databox_2

In the above, we can see that we have promoted our keys from different values
all of type `std::string` to different types entirely. We are not limited to
fundamental types, we are free to make our own structs that serve as keys.
These user-created types are called *tags*.

As the sole purpose of the tag is to provide the compiler with a type
distinguishable from other types, they can be as simple as the following:

\snippet Test_DataBoxDocumentation.cpp std_tuple_tags

Note that we have now promoted `Velocity`, `Radius`, etc. from being *values*
associated with `std::string`s at run time, to *types* distinguishable from
other types at compile time. A large portion of SpECTRE is designed with the
philosophy of enlisting the help of the compiler in assuring the correctness
of our programs. An example of a `std::tuple` making use of these tags might
look like:

~~~{.c}
// Note: This won't work!
std::tuple<Velocity, Radius, Density, Volume> sophomore_databox =
  std::make_tuple(4.0, 2.0, 0.5, 10.0);
~~~

Unfortunately, this will not work. The types passed as template parameters to
`std::tuple` must also be the types of the arguments passed to
`std::make_tuple`. Using a `std::pair`, we could write the above as:

\snippet Test_DataBoxDocumentation.cpp std_tuple_small_program_1

What remains is to rewrite our functions to use `std::tuple` instead of
`std::map`. Note that since we are now using a heterogeneous container of
potentially unknown type, our functions must be templated on the
pairs used to create the `sophomore_databox`. Our functions then look like:

\snippet Test_DataBoxDocumentation.cpp std_tuple_mass_compute

\snippet Test_DataBoxDocumentation.cpp std_tuple_acceleration_compute

\snippet Test_DataBoxDocumentation.cpp std_tuple_force_compute

Using all these `std::pair`s to get our `std::tuple` to work is a bit
cumbersome. There is another way to package together the tagging ability
of the struct names with the type information of the values we wish to store.
To do this we need to make modifications to both our tags as well as our
%Databox implementation. This is what is done in SpECTRE's
`tuples::TaggedTuple`, which is an improved implementation of `std::tuple` in
terms of both performance and interface.

##A TaggedTuple DataBox {#a_taggedtuple_databox}

TaggedTuple is an implementation of a compile time container where the keys
are tags.

Tags that are compatible with SpECTRE's `tuples::TaggedTuple` must have the
type alias `type` in their structs. This type alias carries the type
information of the data we wish to store in the databox. `tuples::TaggedTuple`
is able to make use of this type information so we won't need auxiliary
constructs such as `std::pair` to package this information together anymore.
Our new tags now look like:

\snippet Test_DataBoxDocumentation.cpp tagged_tuple_tags

We are now able to create the `junior_databox` below in the same way we
initially wished to create the `sophomore_databox` above:

\snippet Test_DataBoxDocumentation.cpp tagged_tuple_databox_1

Our functions similarly simplify:

\snippet Test_DataBoxDocumentation.cpp tagged_tuple_mass_compute

\snippet Test_DataBoxDocumentation.cpp tagged_tuple_acceleration_compute

\snippet Test_DataBoxDocumentation.cpp tagged_tuple_force_compute

In each of these iterations of the Databox, we started with initial quantities
and computed subsequent quantities. Let us consider again `force_compute`, in
which `mass` and `acceleration` are recomputed for every call to
`force_compute`. If `Mass` and `Acceleration` were tags somewhow, that is, if we
could compute them once, place them in the databox, and get them back out
through the use of tags, we could get around this problem. We are now ready to
consider SpECTRE's DataBox, which provides the solution to this problem in the
form of `ComputeTags`.

# SpECTRE's DataBox {#a_proper_databox}

A brief description of SpECTRE's DataBox: a TaggedTuple with compute-on-demand.
For a detailed description of SpECTRE's DataBox, see the
\ref DataBoxGroup "DataBox documentation".

## SimpleTags {#documentation_for_simple_tags}
Just as we needed to modify our tags to make them compatible with
`TaggedTuple`, we need to again modify them for use with DataBox. Our ordinary
tags become SpECTRE's SimpleTags:

\snippet Test_DataBoxDocumentation.cpp proper_databox_tags

As seen above, SimpleTags have a `type` and a `name` in their struct.
When creating tags for use with a DataBox, we must make sure to the tag
inherits from one of the existing DataBox tag types such as `db::SimpleTag`.
We now create our first DataBox using these `SimpleTags`:

\snippet Test_DataBoxDocumentation.cpp refined_databox

We can get our quantities out of the DataBox by using `db::get`:

\snippet Test_DataBoxDocumentation.cpp refined_databox_get

So far, the usage of DataBox has been similar to the usage of TaggedTuple. To
address the desire to combine the functionality of tags with the modularity of
functions, DataBox provides ComputeTags.

## ComputeTags {#documentation_for_compute_tags}
ComputeTags are used to tag functions that are used in conjunction with a
DataBox to produce a new quantity. ComputeTags look like:

\snippet Test_DataBoxDocumentation.cpp compute_tags

ComputeTags inherit from `db::ComputeTag`, and it is convenient to have them
additionally inherit from an existing SimpleTag (in this case `Mass`) so that
the quantity `MassCompute` can be obtained through the SimpleTag `Mass`.
We use the naming convention `TagNameCompute` so that `TagNameCompute` and
`TagName` appear next to each other in documentation that lists tags in
alphabetical order.

We have also added the type alias `argument_tags`, which is necessary in order
to refer to the correct tagged quantities in the DataBox.

\note
The `tmpl::list` used in the type alias is a contiguous container only
holding types. That is, there is no variable runtime data associated with it
like there is for `std::tuple`, which is a container associating types with
values. `tmpl::list`s are useful in situations when one is working with
multiple tags at once.

Using nested type aliases to pass around information at compile time is a
common pattern in SpECTRE. Let us see how we can compute our beloved quantity
of mass times acceleration:

\snippet Test_DataBoxDocumentation.cpp compute_tags_force_compute

And that's it! `db::get` utilizes the `argument_tags` specified in
`ForceCompute` to determine which items to get out of the `refined_databox`.
With the corresponding quantities in hand, `db::get` passes them as arguments
to the `function` specified in `ForceCompute`. This is why every `ComputeTag`
must have an `argument_tags` as well as a `function` specified; this is the
contract with DataBox they must satisfy in order to enjoy the full benefits of
DataBox's generality.

## Mutating DataBox items {#documentation_for_mutate_tags}
It is reasonable to expect that in a complicated calculation, we will encounter
time-dependent or iteration-dependent variables. As a result, in addition to
adding and retrieving items from our DataBox, we also need a way to *mutate*
quantities already present in the DataBox. This can be done via
`db::mutate_apply` and can look like the following:

\note
There is an alternative to `db::mutate_apply`, `db::mutate`. See the
\ref DataBoxGroup "DataBox documentation" for more details.

\snippet Test_DataBoxDocumentation.cpp mutate_tags
\snippet Test_DataBoxDocumentation.cpp time_dep_databox

\note
The `not_null`s here are used to give us the assurance that the pointers
`time` and `falling_speed` are not null pointers. Using raw pointers alone, we
risk running into segmentation faults if we dereference null pointers. With
`not_null`s we instead run into a run time error that tells us what went wrong.
For information on the usage of `not_null`, see the documentation for Gsl.hpp.

In the above `db::mutate_apply` example, we are changing two values in the
DataBox using four values from the DataBox. The mutated quantities must be
passed in as `gsl::not_null`s to the lambda. The non-mutated quantities are
passed in as const references to the lambda.

\note
It is critical to guarantee that there is a strict demarcation between
pre-`db::mutate` and post-`db::mutate` quantities. `db::mutate` provides this
guarantee via a locking mechanism; within one mutate call, all initial
pre-mutated quantities are obtained from the DataBox before performing a single
mutation.

\note
The mutate functions described above are the only accepted ways to edit data
in the Databox. It is technically possible to use pointers or references to
edit data stored in the Databox, but this bypasses the compute tags
architecture. All changes to the Databox must be made by the Databox itself
via mutate functions.

From the above, we can see that the different kinds of tags are provided in two
different `tmpl::list`s. The `MutateTags`, also called `ReturnTags`, refer to
the quantities in the DataBox we wish to mutate, and the `ArgumentTags` refer to
additional quantities we need from the DataBox to complete our computation. We
now return to the recurring question of how to make this construction more
modular.

We have now worked our way up to SpECTRE's DataBox, but as we can see in the
above `db::mutate_apply` example, the lambda used to perform the mutation ends
up being independent of any tags or template parameters! This means we can
factor it out and place it in its own module, where it can be tested
independently of the DataBox.

# Toward SpECTRE's Actions {#towards_actions}

## Mutators {#documentation_for_mutators}
These constructs that exist independently of the DataBox are the precursors to
SpECTRE's *Actions*. As they are designed to be used with `db::mutate` and
`db::mutate_apply`, we give them the name *Mutators*. Here is the above lambda
written as a Mutator-prototype, a struct-with-void-apply:

\snippet Test_DataBoxDocumentation.cpp intended_mutation

The call to `db::mutate_apply` has now been made much simpler:

\snippet Test_DataBoxDocumentation.cpp time_dep_databox2

There is a key step that we take here after this point, to make our
struct-with-void-apply into a proper Mutator. As we will see, this addition
will allow us to entirely divorce the internal details of `IntendedMutation`
from the mechanism through which we update the DataBox. The key step is to
add type aliases to `IntendedMutation`:

\snippet Test_DataBoxDocumentation.cpp intended_mutation2

We are now able to write our call to `db::mutate_apply` in the following way:

\snippet Test_DataBoxDocumentation.cpp time_dep_databox3

As we saw earlier with `QuantityCompute`, we found that we were able to imbue
structs with the ability to "read in" types specified in other structs, through
the use of templates and member type aliases. This liberated us from having to
hard-code in specific types. We notice immediately that `IntendedMutation` is a
hard-coded type that we can factor out in favor of a template parameter:

\snippet Test_DataBoxDocumentation.cpp my_first_action

Note how the `return_tags` and `argument_tags` are used as metavariables and
are resolved by the compiler. Our call to `db::mutate_apply` has been fully
wrapped and now takes the form:

\snippet Test_DataBoxDocumentation.cpp time_dep_databox4

The details of applying Mutators to the DataBox are entirely handled by
`MyFirstAction`, with the details of the specific Mutator itself entirely
encapsulated in `IntendedMutation`.

SpECTRE algorithms are decomposed into Actions which can depend on more
things than we have considered here. Feel free to look at the
existing \ref ActionsGroup "actions that have been written." The intricacies of
Actions at the level that SpECTRE uses them is the subject of a future addition
to the \ref dev_guide "Developer's Guide."
