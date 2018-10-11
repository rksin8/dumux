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
 * \ingroup Discretization
 * \brief Declares properties required for finite-volume models models.
 */

#ifndef DUMUX_FV_PROPERTIES_HH
#define DUMUX_FV_PROPERTIES_HH

#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>

#include <dumux/common/properties.hh>
#include <dumux/common/properties/grid.hh>
#include <dumux/common/boundarytypes.hh>

#include <dumux/discretization/fvgridvariables.hh>

namespace Dumux {
namespace Properties {

//! Type tag for finite-volume schemes.
NEW_TYPE_TAG(FiniteVolumeModel, INHERITS_FROM(GridProperties));

//! The grid variables
SET_PROP(FiniteVolumeModel, GridVariables)
{
private:
    using GG = typename GET_PROP_TYPE(TypeTag, FVGridGeometry);
    using GVV = typename GET_PROP_TYPE(TypeTag, GridVolumeVariables);
    using GFVC = typename GET_PROP_TYPE(TypeTag, GridFluxVariablesCache);
public:
    using type = FVGridVariables<GG, GVV, GFVC>;
};

//! We do not store the FVGeometry by default
SET_BOOL_PROP(FiniteVolumeModel, EnableFVGridGeometryCache, false);

//! We do not store the volume variables by default
SET_BOOL_PROP(FiniteVolumeModel, EnableGridVolumeVariablesCache, false);

//! disable flux variables data caching by default
SET_BOOL_PROP(FiniteVolumeModel, EnableGridFluxVariablesCache, false);

//! Boundary types at a single degree of freedom
SET_TYPE_PROP(FiniteVolumeModel, BoundaryTypes, BoundaryTypes<GET_PROP_TYPE(TypeTag, ModelTraits)::numEq()>);

// TODO: bundle SolutionVector, JacobianMatrix
//       in LinearAlgebra traits

//! The type of a solution for the whole grid at a fixed time TODO: move to LinearAlgebra traits
SET_TYPE_PROP(FiniteVolumeModel, SolutionVector, Dune::BlockVector<typename GET_PROP_TYPE(TypeTag, PrimaryVariables)>);

//! Set the type of a global jacobian matrix from the solution types TODO: move to LinearAlgebra traits
SET_PROP(FiniteVolumeModel, JacobianMatrix)
{
private:
    using Scalar = typename GET_PROP_TYPE(TypeTag, Scalar);
    enum { numEq = GET_PROP_TYPE(TypeTag, ModelTraits)::numEq() };
    using MatrixBlock = typename Dune::FieldMatrix<Scalar, numEq, numEq>;
public:
    using type = typename Dune::BCRSMatrix<MatrixBlock>;
};

} // namespace Properties
} // namespace Dumux

 #endif