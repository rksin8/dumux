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
 * \brief Test for the elasticity 1p2c model
 */
#include <config.h>

#include "el1p2cproblem.hh"
#include <dune/common/precision.hh>
#include <dumux/common/start.hh>

/*!
 * \brief Provides an interface for customizing error messages associated with
 *        reading in parameters.
 *
 * \param progName  The name of the program, that was tried to be started.
 * \param errorMsg  The error message that was issued by the start function.
 *                  Comprises the thing that went wrong and a general help message.
 */
void usage(const char *progName, const std::string &errorMsg)
{
    if (errorMsg.size() > 0) {
        std::string errorMessageOut = "\nUsage: ";
                    errorMessageOut += progName;
                    errorMessageOut += " [options]\n";
                    errorMessageOut += errorMsg;
                    errorMessageOut += "\n\nThe List of Mandatory arguments for this program is:\n"
                                        "\t-tEnd                          The end of the simulation. [s] \n"
                                        "\t-dtInitial                     The initial timestep size. [s] \n"
                                        "\t-gridFile                      The file name of the file containing the grid \n"
                                        "\t                                   definition in DGF format\n"
                                        "\t-FluidSystem.nTemperature      Number of tabularization entries [-] \n"
                                        "\t-FluidSystem.nPressure         Number of tabularization entries [-] \n"
                                        "\t-FluidSystem.pressureLow       Low end for tabularization of fluid properties [Pa] \n"
                                        "\t-FluidSystem.pressureHigh      High end for tabularization of fluid properties [Pa] \n"
                                        "\t-FluidSystem.temperatureLow    Low end for tabularization of fluid properties [Pa] \n"
                                        "\t-FluidSystem.temperatureHigh   High end for tabularization of fluid properties [Pa] \n"
                                        "\t-SimulationControl.name        The name of the output files [-] \n"
                                        "\t-InitialConditions.temperature Initial temperature in the reservoir [K] \n"
                                        "\t-InitialConditions.depthBOR    Depth below ground surface [m] \n";

        std::cout << errorMessageOut
                  << "\n";
    }
}

#include <iostream>

int main(int argc, char** argv)
{
    Dune::FMatrixPrecision<>::set_singular_limit(1e-22);
    typedef TTAG(El1P2CProblem) ProblemTypeTag;
    return Dumux::start<ProblemTypeTag>(argc, argv, usage);
}
