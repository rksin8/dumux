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
 * \brief test for the tracer CC model
 */
#include <config.h>

#include "1ptestproblem.hh"
#include "tracertestproblem.hh"

#include <ctime>
#include <iostream>

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/timer.hh>
#include <dune/grid/io/file/dgfparser/dgfexception.hh>
#include <dune/grid/io/file/vtk.hh>
#include <dune/istl/io.hh>

#include <dumux/common/propertysystem.hh>
#include <dumux/common/parameters.hh>
#include <dumux/common/valgrind.hh>
#include <dumux/common/dumuxmessage.hh>
#include <dumux/common/defaultusagemessage.hh>
#include <dumux/common/parameterparser.hh>

#include <dumux/linear/seqsolverbackend.hh>
#include <dumux/nonlinear/newtonmethod.hh>

#include <dumux/assembly/ccassembler.hh>
#include <dumux/assembly/diffmethod.hh>

#include <dumux/io/vtkoutputmodule.hh>

int main(int argc, char** argv)
{
    using namespace Dumux;

    //! define the type tags for this problem
    using OnePTypeTag = TTAG(IncompressibleTestProblem);
    using TracerTypeTag = TTAG(TracerTestCCProblem);

    //! initialize MPI, finalize is done automatically on exit
    const auto& mpiHelper = Dune::MPIHelper::instance(argc, argv);

    //! print dumux start message
    if (mpiHelper.rank() == 0)
        DumuxMessage::print(/*firstCall=*/true);

    ////////////////////////////////////////////////////////////
    // parse the command line arguments and input file
    ////////////////////////////////////////////////////////////

    //! parse command line arguments
    using OnePParameterTree = typename GET_PROP(OnePTypeTag, ParameterTree);
    ParameterParser::parseCommandLineArguments(argc, argv, OnePParameterTree::tree());

    using TracerParameterTree = typename GET_PROP(TracerTypeTag, ParameterTree);
    ParameterParser::parseCommandLineArguments(argc, argv, TracerParameterTree::tree());

    //! parse the input file into the parameter tree
    ParameterParser::parseInputFile(argc, argv, OnePParameterTree::tree(), "1p.input");
    ParameterParser::parseInputFile(argc, argv, TracerParameterTree::tree(), "tracer.input");

    //////////////////////////////////////////////////////////////////////
    // try to create a grid (from the given grid file or the input file)
    /////////////////////////////////////////////////////////////////////

    // only create the grid once using the 1p type tag
    using GridCreator = typename GET_PROP_TYPE(OnePTypeTag, GridCreator);
    try { GridCreator::makeGrid(); }
    catch (...) {
        std::cout << "\n\t -> Creation of the grid failed! <- \n\n";
        throw;
    }
    GridCreator::loadBalance();

    //! we compute on the leaf grid view
    const auto& leafGridView = GridCreator::grid().leafGridView();

    ////////////////////////////////////////////////////////////
    // setup & solve 1p problem on this grid
    ////////////////////////////////////////////////////////////

    //! create the finite volume grid geometry
    using OnePFVGridGeometry = typename GET_PROP_TYPE(OnePTypeTag, FVGridGeometry);
    auto onePFvGridGeometry = std::make_shared<OnePFVGridGeometry>(leafGridView);
    onePFvGridGeometry->update();

    //! the problem (boundary conditions)
    using OnePProblem = typename GET_PROP_TYPE(OnePTypeTag, Problem);
    auto problemOneP = std::make_shared<OnePProblem>(onePFvGridGeometry);

    //! the solution vector
    using SolutionVector = typename GET_PROP_TYPE(OnePTypeTag, SolutionVector);
    SolutionVector p(leafGridView.size(0));

    //! the grid variables
    using OnePGridVariables = typename GET_PROP_TYPE(OnePTypeTag, GridVariables);
    auto onePGridVariables = std::make_shared<OnePGridVariables>(problemOneP, onePFvGridGeometry);
    onePGridVariables->init(p);

    //! the assembler
    using OnePAssembler = CCAssembler<OnePTypeTag, DiffMethod::numeric>;
    auto assemblerOneP = std::make_shared<OnePAssembler>(problemOneP, onePFvGridGeometry, onePGridVariables);

    //! the linear solver
    using LinearSolver = UMFPackBackend<OnePTypeTag>;
    auto linearSolver = std::make_shared<LinearSolver>();

    //! the non-linear solver
    using NewtonController = typename GET_PROP_TYPE(OnePTypeTag, NewtonController);
    auto newtonController = std::make_shared<NewtonController>(leafGridView.comm());
    using Newton = NewtonMethod<OnePTypeTag, NewtonController, OnePAssembler, LinearSolver>;
    Newton nonLinearSolver(newtonController, assemblerOneP, linearSolver);

    //! solve the 1p problem
    nonLinearSolver.solve(p);

    //! write output to vtk
    using GridView = typename GET_PROP_TYPE(OnePTypeTag, GridView);
    Dune::VTKWriter<GridView> onepWriter(leafGridView);
    onepWriter.addCellData(p, "pressure");
    const auto& k = problemOneP->spatialParams().getKField();
    onepWriter.addCellData(k, "permeability");
    onepWriter.write("1p");

    ////////////////////////////////////////////////////////////
    // compute volume fluxes for the tracer model
    ////////////////////////////////////////////////////////////
    using Scalar =  typename GET_PROP_TYPE(OnePTypeTag, Scalar);
    std::vector<Scalar> volumeFlux(onePFvGridGeometry->numScvf(), 0.0);

    using FluxVariables =  typename GET_PROP_TYPE(OnePTypeTag, FluxVariables);
    auto upwindTerm = [](const auto& volVars) { return volVars.mobility(0); };
    for (const auto& element : elements(leafGridView))
    {
        auto fvGeometry = localView(*onePFvGridGeometry);
        fvGeometry.bind(element);

        auto elemVolVars = localView(onePGridVariables->curGridVolVars());
        elemVolVars.bind(element, fvGeometry, p);

        auto elemFluxVars = localView(onePGridVariables->gridFluxVarsCache());
        elemFluxVars.bind(element, fvGeometry, elemVolVars);

        for (const auto& scvf : scvfs(fvGeometry))
        {
            const auto idx = scvf.index();

            if (!scvf.boundary())
            {
                FluxVariables fluxVars;
                fluxVars.init(*problemOneP, element, fvGeometry, elemVolVars, scvf, elemFluxVars);
                volumeFlux[idx] = fluxVars.advectiveFlux(0, upwindTerm);
            }
            else
            {
                const auto bcTypes = problemOneP->boundaryTypes(element, scvf);
                if (bcTypes.hasOnlyDirichlet())
                {
                    FluxVariables fluxVars;
                    fluxVars.init(*problemOneP, element, fvGeometry, elemVolVars, scvf, elemFluxVars);
                    volumeFlux[idx] = fluxVars.advectiveFlux(0, upwindTerm);
                }
            }
        }
    }

    ////////////////////////////////////////////////////////////
    // setup & solve tracer problem on the same grid
    ////////////////////////////////////////////////////////////

    //! create the finite volume grid geometry
    using TracerFVGridGeometry = typename GET_PROP_TYPE(TracerTypeTag, FVGridGeometry);
    auto fvGridGeometry = std::make_shared<TracerFVGridGeometry>(leafGridView);
    fvGridGeometry->update();

    //! the problem (initial and boundary conditions)
    using TracerProblem = typename GET_PROP_TYPE(TracerTypeTag, Problem);
    auto tracerProblem = std::make_shared<TracerProblem>(fvGridGeometry);

    // set the flux from the 1p problem
    tracerProblem->spatialParams().setVolumeFlux(volumeFlux);

    //! the solution vector
    SolutionVector x(leafGridView.size(0));
    tracerProblem->applyInitialSolution(x);
    auto xOld = x;

    //! the grid variables
    using GridVariables = typename GET_PROP_TYPE(TracerTypeTag, GridVariables);
    auto gridVariables = std::make_shared<GridVariables>(tracerProblem, fvGridGeometry);
    gridVariables->init(x, xOld);

    //! get some time loop parameters
    auto tEnd = GET_RUNTIME_PARAM_FROM_GROUP(TracerTypeTag, Scalar, TimeLoop, TEnd);
    auto dt = GET_RUNTIME_PARAM_FROM_GROUP(TracerTypeTag, Scalar, TimeLoop, DtInitial);
    auto maxDt = GET_PARAM_FROM_GROUP(TracerTypeTag, Scalar, TimeLoop, MaxTimeStepSize);

    //! instantiate time loop
    auto timeLoop = std::make_shared<CheckPointTimeLoop<Scalar>>(0.0, dt, tEnd);
    timeLoop->setMaxTimeStepSize(maxDt);

    //! the assembler with time loop for instationary problem
    using TracerAssembler = CCAssembler<TracerTypeTag, DiffMethod::numeric, /*implicit=*/false>;
    auto assembler = std::make_shared<TracerAssembler>(tracerProblem, fvGridGeometry, gridVariables, timeLoop);
    using JacobianMatrix = typename GET_PROP_TYPE(TracerTypeTag, JacobianMatrix);
    auto A = std::make_shared<JacobianMatrix>();
    auto r = std::make_shared<SolutionVector>();
    assembler->setLinearSystem(A, r);

    //! intialize the vtk output module
    VtkOutputModule<TracerTypeTag> vtkWriter(*tracerProblem, *fvGridGeometry, *gridVariables, x, tracerProblem->name());
    using VtkOutputFields = typename GET_PROP_TYPE(TracerTypeTag, VtkOutputFields);
    VtkOutputFields::init(vtkWriter); //! Add model specific output fields
    vtkWriter.write(0.0);

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // run instationary non-linear simulation
    /////////////////////////////////////////////////////////////////////////////////////////////////

    //! set some check points for the time loop
    timeLoop->setPeriodicCheckPoint(tEnd/10.0);

    //! start the time loop
    timeLoop->start(); do
    {
        // set previous solution for storage evaluations
        assembler->setPreviousSolution(xOld);

        Dune::Timer assembleTimer;
        assembler->assembleJacobianAndResidual(x);
        assembleTimer.stop();

        // solve the linear system A(xOld-xNew) = r
        Dune::Timer solveTimer;
        SolutionVector xDelta(x);
        linearSolver->solve(*A, xDelta, *r);
        solveTimer.stop();

        // update solution and grid variables
        Dune::Timer updateTimer;
        x -= xDelta;
        gridVariables->update(x);
        updateTimer.stop();

        // statistics
        const auto elapsedTot = assembleTimer.elapsed() + solveTimer.elapsed() + updateTimer.elapsed();
        std::cout << "Assemble/solve/update time: "
                  <<  assembleTimer.elapsed() << "(" << 100*assembleTimer.elapsed()/elapsedTot << "%)/"
                  <<  solveTimer.elapsed() << "(" << 100*solveTimer.elapsed()/elapsedTot << "%)/"
                  <<  updateTimer.elapsed() << "(" << 100*updateTimer.elapsed()/elapsedTot << "%)"
                  <<  std::endl;

        // make the new solution the old solution
        xOld = x;
        gridVariables->advanceTimeStep();

        // advance to the time loop to the next step
        timeLoop->advanceTimeStep();

        // write vtk output on check points
        if (timeLoop->isCheckPoint())
            vtkWriter.write(timeLoop->time());

        // report statistics of this time step
        timeLoop->reportTimeStep();

        // set new dt as suggested by newton controller
        timeLoop->setTimeStepSize(dt);

    } while (!timeLoop->finished());

    timeLoop->finalize(leafGridView.comm());

    ////////////////////////////////////////////////////////////
    // finalize, print dumux message to say goodbye
    ////////////////////////////////////////////////////////////

    //! print dumux end message
    if (mpiHelper.rank() == 0)
        DumuxMessage::print(/*firstCall=*/false);

    return 0;

} // end main
