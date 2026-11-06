\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Estimate pi with Monte Carlo integration {#tutorial_pi_monte_carlo}

\tableofcontents

This tutorial builds a small parallel executable that estimates pi. Along the
way, it introduces input-file options, DataBox tags, iterable and reduction
actions, array and singleton parallel components, and the metavariables that
connect those pieces. Read \ref tutorial_parallel_concepts first if these
terms are new to you.

The tutorial is based on
[Monte Carlo integration](https://en.wikipedia.org/wiki/Monte_Carlo_integration),
which estimates an integral by averaging samples drawn at random. Consider the
unit square and the quarter unit disk inside it. The square has area one and
the quarter disk has area pi divided by four, so

\f[
  \frac{\pi}{4}
  = \int_0^1 \int_0^1
      \mathbf{1}_{x^2 + y^2 < 1}\,\mathrm{d}x\,\mathrm{d}y.
\f]

Draw \f$N\f$ points uniformly from the square and let \f$H\f$ be the number for
which \f$x^2+y^2<1\f$. The fraction \f$H/N\f$ approaches the area of the quarter
disk as the sample grows, giving the estimator

\f[
  \widehat{\pi} = 4\frac{H}{N}.
\f]

\image html MonteCarloPi.png "Uniform samples inside (blue) and outside (orange) the quarter unit disk."

The executable distributes this sampling work over the available processing
elements (PEs). One `DartThrower` array element on each PE generates samples.
The elements contribute their hit counts to a reduction, and a singleton
`PiEstimator` receives the sum, updates the cumulative estimate, and either
requests another iteration or lets the executable exit.

\note The example compares its estimate with the known value of pi to decide
when to stop. This makes the control flow easy to see, but a real Monte Carlo
integrator would usually estimate its uncertainty from the samples instead of
comparing with an answer known in advance.

## Executable design and communication pattern
Before building the executable, let's look at the overall design.
The executable holds an array parallel component whose elements
throw darts and a singleton parallel component that collects results from
each element of the array component, processes them, and decides whether to
throw more darts or not, based on the accuracy of the estimate of pi so far.
Each array component element and each parallel component get assigned to
a cpu core when the code runs. Each core also has access to the `GlobalCache`.

\image html MonteCarloPiComponents.jpg "Executable design."

Here is an illustration showing the communication pattern of the executable.
Array component elements send how many of the darts they threw hit the
unit quarter circle. The `PiEstimator` singleton handles the reduction, which
combines data from each element (by addition in this case) to compute one
overall estimate of pi. If the singleton determines the accuracy is not
yet at the accuracy goal, it asks each element of the array component to
throw some more darts (i.e., to sample some more random points).

\image html MonteCarloPiCommuniation.jpg "Executable communication pattern."

## Prepare the exercise {#tutorial_pi_monte_carlo_prepare}

In the commands below, `SPECTRE_ROOT` is the source directory and
`SPECTRE_BUILD_DIR` is an already configured build directory. The starter and
solution are both in
`src/Executables/Examples/PiMonteCarlo`. Start from a clean checkout of these
files, then save the solution and replace the source used by the regular build
target with the starter:

``` shell
cd "$SPECTRE_ROOT/src/Executables/Examples/PiMonteCarlo"
mv PiMonteCarlo.cpp PiMonteCarloSolution.cpp
cp PiMonteCarloStarter.cpp PiMonteCarlo.cpp
```

Build and run the starter:

``` shell
cmake --build "$SPECTRE_BUILD_DIR" --target PiMonteCarlo -j 2
"$SPECTRE_BUILD_DIR/bin/PiMonteCarlo" +p1
```

The starter has an empty component list, so it needs no input file and exits
without doing any work. This is intentional: it keeps every intermediate
version runnable while you define types and actions that are not connected to
the executable yet.

Use the two commands above as a checkpoint after **every numbered step through
step 3.4**. Each intermediate version compiles and exits successfully. You may
see unused-variable warnings while an action is only partly implemented; they
disappear when the later lines use those variables. Starting with step 4.1,
use the YAML-driven checkpoint shown in that step instead.

## Part 0: Define input-file options {#tutorial_pi_monte_carlo_options}

Input-file option tags describe values that the option parser should read.
The example lets a user choose the number of samples generated per PE in one
iteration and the fractional accuracy goal.

### Step 0.0

At `TUTORIAL STEP 0.0`, add these two option tags. The lower bounds prevent an
empty sample, which would lead to division by zero, and a nonpositive accuracy
goal, which would never terminate the iteration.

``` cpp
struct DartsPerIteration {
  using type = size_t;
  static constexpr Options::String help{
      "How many darts to throw on each processor in one iteration"};
  static type lower_bound() { return 1; }
};

struct AccuracyGoal {
  using type = double;
  static constexpr Options::String help{
      "Fractional accuracy goal for pi Monte Carlo estimate"};
  static type lower_bound() { return std::numeric_limits<type>::epsilon(); }
};
```

These types describe parsing but do not store values in a parallel component.
That is the job of DataBox tags.

## Part 1: Define DataBox tags {#tutorial_pi_monte_carlo_databox}

Each parallel component owns a DataBox. Simple tags name and describe the
objects stored in it. See \ref databox_foundations for a broader introduction.

### Step 1.0

At `TUTORIAL STEP 1.0`, add tags that initialize their values from the option
tags in step 0.0:

``` cpp
struct DartsPerIteration : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::DartsPerIteration>;
  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(const size_t& darts_per_iteration) {
    return darts_per_iteration;
  }
};

struct AccuracyGoal : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::AccuracyGoal>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double& accuracy_goal) {
    return accuracy_goal;
  }
};
```

### Step 1.1

The estimator also needs cumulative counters. At `TUTORIAL STEP 1.1`, add two
tags initialized to zero without input-file options:

``` cpp
struct ThrowsAllProcs : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<>;
  static constexpr bool pass_metavariables = false;
  static size_t create_from_options() { return 0; }
};

struct HitsAllProcs : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<>;
  static constexpr bool pass_metavariables = false;
  static size_t create_from_options() { return 0; }
};
```

The `DartThrower` components will store only `DartsPerIteration`. The
`PiEstimator` will store all four tags so it can accumulate results and decide
when to stop.

## Part 2: Implement the actions {#tutorial_pi_monte_carlo_actions}

The starter provides the action signatures. First complete the iterable action
that runs independently on every `DartThrower`, then the reduction action that
runs on the `PiEstimator`.

### Step 2.0

At `TUTORIAL STEP 2.0`, replace the temporary `(void)box` line with a DataBox
lookup:

``` cpp
const auto number_of_darts = db::get<Tags::DartsPerIteration>(box);
```

### Step 2.1

Generate points uniformly in the unit square and count the ones inside the
quarter disk:

``` cpp
std::random_device device;
std::mt19937_64 generator(device());
std::uniform_real_distribution distribution{0.0, 1.0};
size_t hits = 0;
for (size_t i = 0; i < number_of_darts; ++i) {
  const auto x = distribution(generator);
  const auto y = distribution(generator);
  if (x * x + y * y < 1.0) {
    hits += 1;
  }
}
```

The engine is local to an action invocation, so different array elements do
not share mutable random-number state.

### Step 2.2

An action communicates with parallel components through proxies stored in the
global cache. Remove the temporary `(void)cache` and `(void)array_index` lines,
then add:

``` cpp
const auto& pi_estimator_proxy =
    Parallel::get_parallel_component<PiEstimator<Metavars>, Metavars>(cache);
const auto& dart_thrower_element_proxy =
    Parallel::get_parallel_component<DartThrower<Metavars>, Metavars>(cache)
        [array_index];
```

The first proxy addresses the singleton. Indexing the array proxy gives a
proxy for the `DartThrower` element running this action.

### Step 2.3

Package the local hit count as reduction data and contribute it. The reduction
uses `funcl::Plus` to sum the values from all array elements before invoking
`ProcessHitsAndThrows` on the singleton.

``` cpp
const Parallel::ReductionData<
    Parallel::ReductionDatum<size_t, funcl::Plus<>>>
    hits_to_send{hits};
Parallel::contribute_to_reduction<Actions::ProcessHitsAndThrows>(
    hits_to_send, dart_thrower_element_proxy, pi_estimator_proxy);
```

The action then returns `Pause`, as already written in the starter. Each array
element remains paused until the estimator explicitly restarts it.

### Step 2.4

The reduction result contains hits from every PE, so the estimator needs the
number of PEs to reconstruct the number of samples generated. Remove the
temporary `(void)cache` line and add:

``` cpp
const auto num_procs = Parallel::number_of_procs<size_t>(cache);
```

### Step 2.5

Read the per-iteration sample count from the estimator's DataBox:

``` cpp
const auto darts_per_iteration = db::get<Tags::DartsPerIteration>(box);
```

### Step 2.6

Remove the temporary `(void)new_hits` line. Replace the empty `db::mutate_apply`
call with this mutation, which updates both cumulative counters together:

``` cpp
db::mutate_apply<tmpl::list<Tags::HitsAllProcs, Tags::ThrowsAllProcs>,
                 tmpl::list<>>(
    [&new_hits, &darts_per_iteration, &num_procs](
        const gsl::not_null<size_t*> hits_all_procs,
        const gsl::not_null<size_t*> throws_all_procs) {
      *hits_all_procs += new_hits;
      *throws_all_procs += darts_per_iteration * num_procs;
    },
    make_not_null(&box));
```

The first typelist contains the tags to mutate. The second would contain
read-only DataBox arguments passed to the lambda; it is empty because this
lambda captures the other values directly.

### Step 2.7

Compute and print the estimate and its fractional difference from pi:

``` cpp
const auto pi_estimate =
    4.0 * static_cast<double>(db::get<Tags::HitsAllProcs>(box)) /
    static_cast<double>(db::get<Tags::ThrowsAllProcs>(box));
const auto fractional_accuracy =
    std::abs(pi_estimate - M_PI) / M_PI;

Parallel::printf("Pi ~ %1.15f (accuracy: %1.15f)\n", pi_estimate,
                 fractional_accuracy);
```

`Parallel::printf` serializes the output so messages from parallel work do not
interleave like ordinary standard-stream output can.

### Step 2.8

If the estimate has not met the goal, get each `DartThrower` proxy and restart
its algorithm:

``` cpp
if (fractional_accuracy > db::get<Tags::AccuracyGoal>(box)) {
  for (size_t i = 0; i < num_procs; ++i) {
    Parallel::get_parallel_component<DartThrower<Metavars>>(cache)[i]
        .perform_algorithm(true);
  }
}
```

When the goal has been met, the singleton sends no restart messages. All array
elements remain paused, Charm++ detects quiescence, and SpECTRE advances to the
exit phase.

## Part 3: Define parallel components {#tutorial_pi_monte_carlo_components}

Actions describe work, but parallel components specify where the actions run
and what data they own.

### Step 3.0

Replace the empty `PiEstimator` definition at `TUTORIAL STEP 3.0` with this
singleton component:

``` cpp
template <typename Metavars>
struct PiEstimator {
  using chare_type = Parallel::Algorithms::Singleton;
  static constexpr bool checkpoint_data = true;
  using metavariables = Metavars;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Execute, tmpl::list<>>>;
  using simple_tags_from_options =
      tmpl::list<Tags::HitsAllProcs, Tags::ThrowsAllProcs,
                 Tags::DartsPerIteration, Tags::AccuracyGoal>;
  static void execute_next_phase(
      Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavars>& global_cache);
};
```

The singleton has no iterable actions of its own. Its
`ProcessHitsAndThrows` reduction action is invoked remotely when a reduction
finishes.

### Step 3.1

At `TUTORIAL STEP 3.1`, define the phase-transition function declared above:

``` cpp
template <typename Metavars>
void PiEstimator<Metavars>::execute_next_phase(
    const Parallel::Phase next_phase,
    const Parallel::CProxy_GlobalCache<Metavars>& global_cache) {
  auto& local_cache = *Parallel::local_branch(global_cache);
  Parallel::get_parallel_component<PiEstimator<Metavars>>(local_cache)
      .start_phase(next_phase);
}
```

### Step 3.2

Replace the empty `DartThrower` definition at `TUTORIAL STEP 3.2`. This is an
array component with integer indices and `ThrowDarts` as its iterable action
in the execute phase.

``` cpp
template <typename Metavars>
struct DartThrower {
  using chare_type = Parallel::Algorithms::Array;
  static constexpr bool checkpoint_data = true;
  using metavariables = Metavars;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<Parallel::Phase::Execute,
                                        tmpl::list<Actions::ThrowDarts>>>;
  using simple_tags_from_options = tmpl::list<Tags::DartsPerIteration>;
  using array_index = int;
  using array_allocation_tags = tmpl::list<>;

  static void execute_next_phase(
      Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavars>& global_cache);
  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavars>& global_cache,
      const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
          initialization_options,
      const tuples::tagged_tuple_from_typelist<array_allocation_tags>&
          array_allocation_options = {},
      const std::unordered_set<size_t>& procs_to_ignore = {});
};
```

### Step 3.3

Define the array component's phase transition at `TUTORIAL STEP 3.3`:

``` cpp
template <typename Metavars>
void DartThrower<Metavars>::execute_next_phase(
    const Parallel::Phase next_phase,
    const Parallel::CProxy_GlobalCache<Metavars>& global_cache) {
  auto& local_cache = *Parallel::local_branch(global_cache);
  Parallel::get_parallel_component<DartThrower<Metavars>>(local_cache)
      .start_phase(next_phase);
}
```

### Step 3.4

Finally, define how the array elements are allocated. This example creates one
element per PE and assigns them round-robin while respecting any PEs that the
resource options ask it to skip.

``` cpp
template <typename Metavars>
void DartThrower<Metavars>::allocate_array(
    Parallel::CProxy_GlobalCache<Metavars>& global_cache,
    const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
        initialization_options,
    const tuples::tagged_tuple_from_typelist<array_allocation_tags>&
    /*array_allocation_options*/,
    const std::unordered_set<size_t>& procs_to_ignore) {
  auto& local_cache = *Parallel::local_branch(global_cache);
  auto& array_proxy =
      Parallel::get_parallel_component<DartThrower<Metavars>>(local_cache);

  size_t which_proc = 0;
  const auto num_procs = Parallel::number_of_procs<size_t>(local_cache);
  const auto number_of_elements = num_procs;

  for (size_t i = 0; i < number_of_elements; ++i) {
    while (procs_to_ignore.find(which_proc) != procs_to_ignore.end()) {
      which_proc = which_proc + 1 == num_procs ? 0 : which_proc + 1;
    }
    array_proxy[i].insert(global_cache, initialization_options, which_proc);
    which_proc = which_proc + 1 == num_procs ? 0 : which_proc + 1;
  }
  array_proxy.doneInserting();
}
```

## Part 4: Connect the executable {#tutorial_pi_monte_carlo_metavariables}

Nothing above has run yet because the temporary `Metavariables::component_list`
is empty. The final part registers both components and changes the checkpoint
from the empty smoke run to the real YAML-driven calculation.

### Step 4.1

Replace the empty component list at `TUTORIAL STEP 4.1`:

``` cpp
using component_list =
    tmpl::list<PiEstimator<Metavariables>, DartThrower<Metavariables>>;
```

The executable now expects the options requested by those components. Build,
check the supplied input file, and run on two PEs:

``` shell
cmake --build "$SPECTRE_BUILD_DIR" --target PiMonteCarlo -j 2
"$SPECTRE_BUILD_DIR/bin/PiMonteCarlo" \
  --input-file \
  "$SPECTRE_ROOT/tests/InputFiles/ExampleExecutables/PiMonteCarlo.yaml" \
  --check-options +p1
"$SPECTRE_BUILD_DIR/bin/PiMonteCarlo" \
  --input-file \
  "$SPECTRE_ROOT/tests/InputFiles/ExampleExecutables/PiMonteCarlo.yaml" \
  +p2
```

The first command should report that the input file parsed successfully. The
second prints one or more estimates and exits after the requested accuracy has
been reached. Because the samples are random, the values and the number of
iterations vary. Output includes a line similar to:

``` plain
Pi ~ 3.148200000000000 (accuracy: 0.002103183683810)
```

### Step 4.2

Replace the placeholder help string at `TUTORIAL STEP 4.2`:

``` cpp
static constexpr Options::String help{
    "Compute pi with Monte Carlo integration"};
```

Repeat the YAML-driven checkpoint from step 4.1. You can also inspect the
completed executable's option help:

``` shell
"$SPECTRE_BUILD_DIR/bin/PiMonteCarlo" --help +p1
```

## Compare or restore the solution {#tutorial_pi_monte_carlo_finish}

Your completed file should now match `PiMonteCarloSolution.cpp` apart from
comments or equivalent formatting. Compare them with:

``` shell
diff -u PiMonteCarloSolution.cpp PiMonteCarlo.cpp
```

To restore the tracked solution while keeping your completed exercise, run:

``` shell
mv PiMonteCarlo.cpp PiMonteCarloCompleted.cpp
mv PiMonteCarloSolution.cpp PiMonteCarlo.cpp
cmake --build "$SPECTRE_BUILD_DIR" --target PiMonteCarlo -j 2
```

You have now assembled a complete task-parallel SpECTRE executable: array
elements perform independent Monte Carlo work, a reduction combines their
results, and a singleton controls iteration and termination.
