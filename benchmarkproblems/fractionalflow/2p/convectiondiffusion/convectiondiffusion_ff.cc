#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <iostream>
#include"dune/common/mpihelper.hh" // An initializer of MPI
#include"dune/common/exceptions.hh" // We use exceptions
#include <dune/grid/sgrid.hh> // load sgrid definition
#include <iostream>
#include <iomanip>
#include <dune/grid/utility/gridtype.hh>
#include <dune/grid/common/gridinfo.hh>
#include <dune/grid/io/file/dgfparser/dgfparser.hh>
#include <dune/grid/io/file/dgfparser/dgfs.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>
#include <dune/istl/io.hh>
#include <dune/common/timer.hh>
#include "dumux/material/properties.hh"
#include "dumux/material/linearlaw_deprecated.hh"
#include "dumux/material/brookscoreylaw_deprecated.hh"
#include "dumux/material/vangenuchtenlaw_deprecated.hh"
#include "dumux/transport/fv/fvtransport_deprecated.hh"
#include "dumux/transport/fv/capillarydiffusion.hh"
#include "dumux/diffusion/fv/fvdiffusion_deprecated.hh"
#include "dumux/diffusion/fv/fvdiffusionvelocity_deprecated.hh"
#include "dumux/diffusion/mimetic/mimeticdiffusion.hh"
#include "dumux/fractionalflow/impes/impes_deprecated.hh"
#include "../problemdefinitions/convectivediffusiontransportproblem.hh"
#include "../problemdefinitions/convectiondiffusiondiffproblem.hh"
#include "../problemdefinitions/buckleyleverettdiffproblem.hh"
#include "dumux/timedisc/timeloop.hh"
#include "dumux/timedisc/rungekuttastep.hh"
#include "dumux/fractionalflow/variableclass.hh"

int main(int argc, char** argv)
{
    try{
        // define the problem dimensions
        const int dim=1;
        typedef double NumberType;
        typedef GridType::ctype ctype;

        Dune::FieldVector<NumberType, dim> Left(0);
        Dune::FieldVector<NumberType, dim> Right(800);
        if (argc != 2) {
            std::cout << "usage: tEnd" << std::endl;
            return 0;
        }
        std::string arg1(argv[1]);
        std::istringstream is1(arg1);
        double tEnd;
        is1 >> tEnd;

        // create a grid object
        typedef Dune::SGrid<dim,dim> GridType;

        // use unitcube from grids
        std::stringstream dgfFileName;
        dgfFileName << "grids/unitcube" << GridType :: dimension << ".dgf";

        // create grid pointer, GridType is defined by gridtype.hh
        Dune::GridPtr<GridType> gridPtr( dgfFileName.str() );

        // grid reference
        GridType& grid = *gridPtr;

        //deffinition of a stretched grid

        Dune::gridinfo(grid);

        // time loop parameters
        const double tStart = 0;
        // const double tEnd = 2.5e9;
        const double cFLFactor = 1;
        // slope limiter parameters
        //    bool reconstruct = true;
        //    double alphaMax = 0.8;

        // IMPES parameters
        int iterFlag = 2;
        int nIter = 500;
        double maxDefect = 1e-5;
        double omega=0.4;

        // plotting parameters
        const char* fileName = "convectivediffusion";
        int modulo = 10;

        Oil oil(0);
        Water water(0);

        //Dune::LinearLaw materialLaw(water,oil,10);
        Dune::BrooksCoreyLaw materialLaw(water, oil,2.0,100);
        //Dune::VanGenuchtenLaw materialLaw(water,oil,3.1257,1);

        typedef Dune::VariableClass<GridType, NumberType> VC;

        VC variables(grid);

        Dune::ConvectionDiffusionTransportProblem<GridType, NumberType, VC> transportProblem(variables,materialLaw,Left,Right);
        Dune::BuckleyLeverettDiffProblem<GridType, NumberType, VC> diffusionProblem(variables,materialLaw,Left,Right);

        typedef Dune::FVTransport<GridType, NumberType, VC> Transport;
        Transport transport(grid, transportProblem, grid.maxLevel());
        //Dune::CapillaryDiffusion<GridType, NumberType> diffPart(diffusionProblem);
        //Transport transport(grid, transportProblem, grid.maxLevel(),diffPart,reconstruct, alphaMax, cFLFactor,true);

        typedef Dune::FVDiffusionVelocity<GridType, NumberType, VC> Diffusion;
        Diffusion diffusion(grid, diffusionProblem, grid.maxLevel());

        typedef Dune::IMPES<GridType, Diffusion, Transport, VC> IMPES;
        IMPES fractionalflow(diffusion, transport, iterFlag, nIter, maxDefect,omega);

        Dune::TimeLoop<GridType, IMPES > timeloop(tStart, tEnd, fileName, modulo, cFLFactor);

        Dune::Timer timer;
        timer.reset();
        timeloop.execute(fractionalflow);
        std::cout << "timeloop.execute took " << timer.elapsed() << " seconds" << std::endl;
        //printvector(std::cout, *fractionalflow, "saturation", "row", 200, 1);

        return 0;
    }
    catch (Dune::Exception &e){
        std::cerr << "Dune reported error: " << e << std::endl;
    }
    catch (...){
        std::cerr << "Unknown exception thrown!" << std::endl;
    }
}

