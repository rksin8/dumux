This tutorial was copied from tests/porousmediumflow/2p/adaptive and restricted to the cell-centered finite volume TPFA discretization scheme.
You need [ALUGrid][0] in order to compile and run it.

# Two-phase flow with infiltration and adaptive grid

## Problem set-up
Soil contamination problem where DNAPL infiltrates a fully water saturated medium.

...


## Infiltration (point source)



## Adaptive grid
### main.cc
The following headers need to be included in the main file:

```C++
#include <dumux/adaptive/adapt.hh>
#include <dumux/adaptive/markelements.hh>
#include <dumux/adaptive/initializationindicator.hh>
#include <dumux/porousmediumflow/2p/griddatatransfer.hh>
#include <dumux/porousmediumflow/2p/gridadaptindicator.hh>
```

The grid adaptation is prepared during the initialization by the following steps:
* instantiate indicator & data transfer, read parameters for indicator (ll. 111-115)
* do initial refinement around sources/BCs (l. 118)
* refine up to the maximum level (ll. 121-137)
* do refinement for the initial conditions using the indicator (ll. 140-144)
* update grid data after adaption (ll. 147-152)

The indicator ... 

During the time loop, the refinement indicator is computed (l. 191) and the respective elements to be refined are marked (ll. 194-196).

In case of grid adaptation, the following updates are necessary (ll. 201-205):
```C++
 xOld = x; //!< Overwrite the old solution with the new (resized & interpolated) one
 ssembler->setJacobianPattern(); //!< Tell the assembler to resize the matrix and set pattern
 assembler->setResidualSize(); //!< Tell the assembler to resize the residual
 gridVariables->updateAfterGridAdaption(x); //!< Initialize the secondary variables to the new (and "new old") solution
 problem->computePointSourceMap(); //!< Update the point source map
```

### problem.hh
...

### params.input
The following parameters in the `[Adaptive]` parameter group determine the grid adaptation behavior:
* `RefineAtDirichletBC`: ...
* `RefineAtFluxBC`: ...
* `MinLevel`: ...
* `MaxLevel`: ...
* `CoarsenTolerance`: ...
* `RefineTolerance`: ...



[0]: https://gitlab.dune-project.org/extensions/dune-alugrid