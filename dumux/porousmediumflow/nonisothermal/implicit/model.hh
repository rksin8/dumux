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
 * \brief TODO doc
 */

#ifndef DUMUX_POROUSMEDIUMFLOW_NONISOTHERMAL_MODEL_HH
#define DUMUX_POROUSMEDIUMFLOW_NONISOTHERMAL_MODEL_HH

// #include <dumux/implicit/properties.hh>

namespace Dumux
{
//! declaration of the implementation
template<class TypeTag, bool EnableEnergy>
class NonIsothermalModelImplementation;

template<class TypeTag>
using NonIsothermalModel = NonIsothermalModelImplementation<TypeTag, GET_PROP_VALUE(TypeTag, EnableEnergyBalance)>;

template<class TypeTag>
class NonIsothermalModelImplementation<TypeTag, false>
{
public:
    template<class VtkOutputModule>
    static void maybeAddTemperature(VtkOutputModule& vtkOutputModule)
    {}
};

template<class TypeTag>
class NonIsothermalModelImplementation<TypeTag, true>
{
    using Indices = typename GET_PROP_TYPE(TypeTag, Indices);

public:
    template<class VtkOutputModule>
    static void maybeAddTemperature(VtkOutputModule& vtkOutputModule)
    {
        // register vtk output field for temperature
        vtkOutputModule.addPrimaryVariable("temperature", Indices::temperatureIdx);
    }
};

} // end namespace Dumux

#endif
