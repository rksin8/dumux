// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*****************************************************************************
 *   See the file COPYING for full copying permissions.                      *
 *                                                                           *
 *   This program is free software: you can redistribute it and/or modify    *
 *   it under the terms of the GNU General Public License as published by    *
 *   the Free Software Foundation, either version 2 of the License, or       *
 *   (at your option) any later version.                                     *
 *                                                                           *
 *   This program is distributed in the hope that it will be useful,         *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the            *
 *   GNU General Public License for more details.                            *
 *                                                                           *
 *   You should have received a copy of the GNU General Public License       *
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.   *
 *****************************************************************************/
/*!
 * \file
 *
 * \brief Tutorial problem for a fully coupled twophase box model.
 */
#ifndef DUMUX_EXERCISE_THREE_PROBLEM_HH
#define DUMUX_EXERCISE_THREE_PROBLEM_HH

// The numerical model
#include <dumux/porousmediumflow/2p2c/implicit/model.hh>

// The base porous media box problem
#include <dumux/porousmediumflow/implicit/problem.hh>

// Spatially dependent parameters
#include "spatialparams.hh"

// The fluid system that is created in this exercise
#include "fluidsystems/h2omycompressiblecomponent.hh"

namespace Dumux{
// Forward declaration of the problem class
template <class TypeTag> class ExerciseThreeProblem;

namespace Properties {
// Create a new type tag for the problem
NEW_TYPE_TAG(ExerciseThreeProblem, INHERITS_FROM(BoxTwoPTwoC, ExerciseThreeSpatialParams));

// Set the "Problem" property
SET_TYPE_PROP(ExerciseThreeProblem, Problem, ExerciseThreeProblem<TypeTag>);

// Set grid and the grid creator to be used
#if HAVE_DUNE_ALUGRID
SET_TYPE_PROP(ExerciseThreeProblem, Grid, Dune::ALUGrid</*dim=*/2, 2, Dune::cube, Dune::nonconforming>);
#elif HAVE_UG
SET_TYPE_PROP(ExerciseThreeProblem, Grid, Dune::UGGrid<2>);
#else
SET_TYPE_PROP(ExerciseThreeProblem, Grid, Dune::YaspGrid<2>);
#endif // HAVE_DUNE_ALUGRID

 // The fluid system property
SET_PROP(ExerciseThreeProblem, FluidSystem)
{
private:
    typedef typename GET_PROP_TYPE(TypeTag, Scalar) Scalar;
public:
    /*!
     * TODO: set the fluid system created in this exercise as the property
     */
    typedef FluidSystems::H2OMyCompressibleComponent<TypeTag, Scalar> type;
};

// Disable gravity
SET_BOOL_PROP(ExerciseThreeProblem, ProblemEnableGravity, true);
}

/*!
 * \ingroup TwoPBoxModel
 *
 * \brief  Tutorial problem for a fully coupled twophase box model.
 */
template <class TypeTag>
class ExerciseThreeProblem : public ImplicitPorousMediaProblem<TypeTag>
{
    typedef ImplicitPorousMediaProblem<TypeTag> ParentType;
    typedef typename GET_PROP_TYPE(TypeTag, Scalar) Scalar;
    typedef typename GET_PROP_TYPE(TypeTag, GridView) GridView;

    // Grid dimension
    enum { dim = GridView::dimension,
           dimWorld = GridView::dimensionworld
    };

    // Types from DUNE-Grid
    typedef typename GridView::template Codim<0>::Entity Element;
    typedef typename GridView::template Codim<dim>::Entity Vertex;
    typedef typename GridView::Intersection Intersection;
    typedef Dune::FieldVector<Scalar, dimWorld> GlobalPosition;

    // Dumux specific types
    typedef typename GET_PROP_TYPE(TypeTag, TimeManager) TimeManager;
    typedef typename GET_PROP_TYPE(TypeTag, Indices) Indices;
    typedef typename GET_PROP_TYPE(TypeTag, PrimaryVariables) PrimaryVariables;
    typedef typename GET_PROP_TYPE(TypeTag, BoundaryTypes) BoundaryTypes;
    typedef typename GET_PROP_TYPE(TypeTag, FVElementGeometry) FVElementGeometry;
    typedef typename GET_PROP_TYPE(TypeTag, FluidSystem) FluidSystem;

public:
    ExerciseThreeProblem(TimeManager &timeManager,
                         const GridView &gridView)
        : ParentType(timeManager, gridView)
        , eps_(3e-6)
    {
#if !(HAVE_DUNE_ALUGRID || HAVE_UG)
      std::cout << "If you want to use simplices instead of cubes, install and use dune-ALUGrid or UGGrid." << std::endl;
#endif // !(HAVE_DUNE_ALUGRID || HAVE_UG)

      // initialize the fluid system
      FluidSystem::init();

      // set the depth of the bottom of the reservoir
      depthBOR_ = this->bBoxMax()[dimWorld-1];
    }

    //! Specifies the problem name. This is used as a prefix for files
    //! generated by the simulation.
    std::string name() const
    {
        static const std::string name = GET_RUNTIME_PARAM_FROM_GROUP(TypeTag, std::string, Problem, Name);
        return name;
    }

    //! Returns true if a restart file should be written.
    bool shouldWriteRestartFile() const
    { return false; }

    //! Returns true if the current solution should be written to disk
    //! as a VTK file
    bool shouldWriteOutput() const
    {
        return this->timeManager().timeStepIndex() > 0 &&
               this->timeManager().timeStepIndex() % 1 == 0;
    }

    //! Returns the temperature within a finite volume. We use constant
    //! 10 degrees Celsius.
    Scalar temperature() const
    { return 283.15; }

    //! Specifies which kind of boundary condition should be used for
    //! which equation for a finite volume on the boundary.
    void boundaryTypes(BoundaryTypes &bcTypes, const Vertex &vertex) const
    {
        const GlobalPosition &globalPos = vertex.geometry().center();

        // Dirichlet conditions on left & right boundary
        if (globalPos[0] < eps_ || globalPos[0] > this->bBoxMax()[0] - eps_)
           bcTypes.setAllDirichlet();
        else // neuman for the remaining boundaries
           bcTypes.setAllNeumann();

    }

    //! Evaluates the Dirichlet boundary conditions for a finite volume
    //! on the grid boundary. Here, the 'values' parameter stores
    //! primary variables.
    void dirichlet(PrimaryVariables &values, const Vertex &vertex) const
    {
        const auto globalPos = vertex.geometry().center();
        values[Indices::pressureIdx] = 200.0e3 + 9.81*1000*(depthBOR_ - globalPos[dimWorld-1]);
        values[Indices::switchIdx] = 0.0; // 0 % oil saturation on left boundary
    }

    //! Evaluates the boundary conditions for a Neumann boundary
    //! segment. Here, the 'values' parameter stores the mass flux in
    //! [kg/(m^2 * s)] in normal direction of each phase. Negative
    //! values mean influx.
    void neumann(PrimaryVariables &values,
                 const Element &element,
                 const FVElementGeometry &fvGeometry,
                 const Intersection &intersection,
                 int scvIdx,
                 int boundaryFaceIdx) const
    {
        const GlobalPosition &globalPos =
            fvGeometry.boundaryFace[boundaryFaceIdx].ipGlobal;
        Scalar up = this->bBoxMax()[dimWorld-1];
        // extraction of oil on the right boundary for approx. 1.e6 seconds
        if (globalPos[dimWorld-1] > up - eps_ && globalPos[0] > 20 && globalPos[0] < 40) {
            // oil outflux of 30 g/(m * s) on the right boundary.
            // we solve for the mole balance, so we have to divide by the molar mass
            values[Indices::contiWEqIdx] = 0;
            values[Indices::contiNEqIdx] = -3e-2/FluidSystem::MyCompressibleComponent::molarMass();
        } else {
            // no-flow on the remaining Neumann-boundaries.
            values[Indices::contiWEqIdx] = 0;
            values[Indices::contiNEqIdx] = 0;
        }
    }

    //! Evaluates the initial value for a control volume. For this
    //! method, the 'values' parameter stores primary variables.
    void initial(PrimaryVariables &values,
                 const Element &element,
                 const FVElementGeometry &fvGeometry,
                 int scvIdx) const
    {
        const auto globalPos = fvGeometry.subContVol[scvIdx].global;
        values[Indices::pressureIdx] = 200.0e3 + 9.81*1000*(depthBOR_ - globalPos[dimWorld-1]); // 200 kPa = 2 bar
        values[Indices::switchIdx] = 0.0;
    }

    //! Evaluates the source term for all phases within a given
    //! sub-control-volume. In this case, the 'values' parameter
    //! stores the rate mass generated or annihilated per volume unit
    //! in [kg / (m^3 * s)]. Positive values mean that mass is created.
    void source(PrimaryVariables &values,
                const Element &element,
                const FVElementGeometry &fvGeometry,
                int scvIdx) const
    {
        values[Indices::contiWEqIdx] = 0.0;
        values[Indices::contiNEqIdx]= 0.0;
    }

    //! Evaluates the initial phase presence
    int initialPhasePresenceAtPos(const GlobalPosition &globalPos) const
    { return Indices::wPhaseOnly; }

private:
    // small epsilon value
    Scalar eps_;

    // depth at the bottom of the reservoir
    Scalar depthBOR_;
};
}

#endif
