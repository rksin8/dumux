// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*****************************************************************************
 *   See the file COPYING for full copying permissions.                      *
 *                                                                           *
 *   This program is free software: you can redistribute it and/or modify    *
 *   it under the terms of the GNU General Public License as published by    *
 *   the Free Software Foundation, either version 3 of the License, or       *
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
 * \brief test for the pore network model grid creator
 */
#include "config.h"

#include <dune/grid/io/file/vtk.hh>

#include <dumux/common/dumuxmessage.hh>
#include <dumux/common/defaultusagemessage.hh>
#include <dumux/common/parameters.hh>
#include <dumux/io/grid/porenetworkgridcreator.hh>


namespace Dumux
{

template<class GridView, class GridData>
std::vector<std::vector<double>> getVertexParams(const GridView& gridView, const GridData& gridData)
{
    const auto someVertex = *(vertices(gridView).begin());
    const auto numVertexParams = gridData.parameters(someVertex).size();
    std::vector<std::vector<double>> result(numVertexParams);

    for(int i = 0; i < result.size(); ++i)
        result[i].resize(gridView.size(1));

    for(const auto& vertex : vertices(gridView))
    {
        const auto vIdx = gridView.indexSet().index(vertex);
        for(int i = 0; i < result.size(); ++i)
            result[i][vIdx] = gridData.parameters(vertex)[i];
    }

    return result;
}

template<class GridView, class GridData>
std::vector<std::vector<double>> getElementParams(const GridView& gridView, const GridData& gridData)
{
    const auto someElement = *(elements(gridView).begin());
    const auto numElementParams = gridData.parameters(someElement).size();
    std::vector<std::vector<double>> result(numElementParams);

    for(int i = 0; i < result.size(); ++i)
        result[i].resize(gridView.size(0));

    for(const auto& element : elements(gridView))
    {
        const auto eIdx = gridView.indexSet().index(element);
        for(int i = 0; i < result.size(); ++i)
            result[i][eIdx] = gridData.parameters(element)[i];
    }

    return result;
}

template<class GridView, class GridData>
void writeToVtk(const std::string& fileName, const GridView& gridView, const GridData& gridData)
{
    using VTKWriter = Dune::VTKWriter<GridView>;
    static const std::array<std::string, 3> poreParameterNames = {"poreRadius", "poreVolume", "poreLabel"};
    static const std::array<std::string, 3> throatParameterNames = {"throatRadius", "throatlength", "throatLabel"};

    const auto vertexData = getVertexParams(gridView, *gridData);
    const auto elementData = getElementParams(gridView, *gridData);

    VTKWriter vtkWriter(gridView);
    for(int i = 0; i < poreParameterNames.size(); ++i)
        vtkWriter.addVertexData(vertexData[i], poreParameterNames[i]);
    for(int i = 0; i < throatParameterNames.size(); ++i)
        vtkWriter.addCellData(elementData[i], throatParameterNames[i]);

    vtkWriter.write(fileName);
}

template<int dimWorld>
void testGeneric(const std::string& name)
{
    using GridManager = Dumux::PoreNetworkGridCreator<dimWorld>;

    // create a random network, write out a dgf file before grid sanitation
    {
        const std::array<std::string, 3> prefix = {"Generic1D", "Generic2D", "Generic3D"};

        GridManager gridManager;
        gridManager.init(prefix[dimWorld-1]);

        const auto gridView = gridManager.grid().leafGridView();
        const auto gridData = gridManager.getGridData();

        writeDgf("dgfGrid.dgf",
                 gridView,
                 *gridData);

        gridManager.sanitizeGrid();

        const std::array<std::string, 3> vtkName = {"generic-1dgrid-", "generic-2dgrid-", "generic-3dgrid-"};
        writeToVtk(vtkName[dimWorld-1] + name, gridView, gridData);
    }

    // use faulty dgf file to create a grid, apply grid sanitation (specified in input file)
    {
        const std::array<std::string, 3> prefix = {"Dgf1D", "Dgf2D", "Dgf3D"};

        GridManager gridManager;
        gridManager.init(prefix[dimWorld-1]);

        const auto gridView = gridManager.grid().leafGridView();
        const auto gridData = gridManager.getGridData();

        const std::array<std::string, 3> vtkName = {"dgf-1dgrid-", "dgf-2dgrid-", "dgf-3dgrid-"};

        writeToVtk(vtkName[dimWorld-1] + name, gridView, gridData);
    }
}


bool testRemoveThroatsOnBoundary()
{
    using GridManager = Dumux::PoreNetworkGridCreator<2>;
    using GridType = typename GridManager::Grid;
    using Element = typename GridType::template Codim<0>::Entity;
    using GlobalPosition = typename Element::Geometry::GlobalCoordinate;

    GridManager gridManager;
    gridManager.init();

    using std::min;
    using std::max;

    const auto gridView = gridManager.grid().leafGridView();
    const auto gridData = gridManager.getGridData();

    if(getParam<bool>("Test.WriteVTK"))
        writeToVtk("remove-throats", gridView, gridData);

    const auto lowerLeft = getParam<GlobalPosition>("Grid.LowerLeft", GlobalPosition(0.0));
    const auto upperRight = getParam<GlobalPosition>("Grid.UpperRight");

    double elemCenterXMin = 1e9;
    double elemCenterXMax = -1e9;
    double elemCenterYMin = 1e9;
    double elemCenterYMax = -1e9;

    for(const auto& element : elements(gridView))
    {
        const auto center = element.geometry().center();
        elemCenterXMin = min(elemCenterXMin, center[0]);
        elemCenterYMin = min(elemCenterYMin, center[1]);
        elemCenterXMax = max(elemCenterXMax, center[0]);
        elemCenterYMax = max(elemCenterYMax, center[1]);
    }

    const auto removeThroatsOnBoundary = getParam<std::vector<unsigned int>>("Grid.RemoveThroatsOnBoundary", std::vector<unsigned int>{});

    const double eps = 1e-8;

    for(int i : removeThroatsOnBoundary)
    {
        switch(i)
        {
            case 0: if(elemCenterXMin < lowerLeft[0] + eps) return false; break;
            case 1: if(elemCenterXMax > upperRight[0] - eps) return false; break;
            case 2: if(elemCenterYMin < lowerLeft[1] + eps) return false; break;
            case 3: if(elemCenterYMax > upperRight[1] - eps) return false; break;
        }
    }

    return true;
}

} // end namespace Dumux

int main(int argc, char** argv) try
{
    using namespace Dumux;

    // initialize MPI, finalize is done automatically on exit
    const auto& mpiHelper = Dune::MPIHelper::instance(argc, argv);

    // print dumux start message
    if (mpiHelper.rank() == 0)
        DumuxMessage::print(/*firstCall=*/true);

    ////////////////////////////////////////////////////////////
    // parse the command line arguments and input file
    ////////////////////////////////////////////////////////////

    // parse command line arguments
    Parameters::init(argc, argv);

    const auto testName = getParam<std::string>("Test.Name");

    // test the functionality for deleting throats on the boundary
    if(testName == "remove-throats-on-boundary")
    {
        const bool testPassed = testRemoveThroatsOnBoundary();
        if(testPassed)
            return 0;
        else
            return 1;
    }

    // test 1D, 2D and 3D grids
    std::cout << " **** Testing generic 3D grid ****\n ";
    testGeneric<3>(testName);

    std::cout << " \n\n**** Testing generic 2D grid ****\n ";
    testGeneric<2>(testName);

    std::cout << " \n\n**** Testing generic 1D grid ****\n ";
    testGeneric<1>(testName);

    ////////////////////////////////////////////////////////////
    // finalize, print dumux message to say goodbye
    ////////////////////////////////////////////////////////////

    // print dumux end message
    if (mpiHelper.rank() == 0)
    {
        Parameters::print();
        DumuxMessage::print(/*firstCall=*/false);
    }

    return 0;
}

catch (Dumux::ParameterException &e)
{
    std::cerr << std::endl << e << " ---> Abort!" << std::endl;
    return 1;
}
catch (Dune::DGFException & e)
{
    std::cerr << "DGF exception thrown (" << e <<
                 "). Most likely, the DGF file name is wrong "
                 "or the DGF file is corrupted, "
                 "e.g. missing hash at end of file or wrong number (dimensions) of entries."
                 << " ---> Abort!" << std::endl;
    return 2;
}
catch (Dune::Exception &e)
{
    std::cerr << "Dune reported error: " << e << " ---> Abort!" << std::endl;
    return 3;
}
catch (...)
{
    std::cerr << "Unknown exception thrown! ---> Abort!" << std::endl;
    return 4;
}
