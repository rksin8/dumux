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
 * \ingroup Discretization
 * \brief free functions for the evaluation of primary variables inside elements.
 */
#ifndef DUMUX_DISCRETIZATION_EVAL_SOLUTION_HH
#define DUMUX_DISCRETIZATION_EVAL_SOLUTION_HH

#include <dune/localfunctions/lagrange/pqkfactory.hh>

#include <dumux/discretization/box/elementsolution.hh>
#include <dumux/discretization/cellcentered/elementsolution.hh>

namespace Dumux
{

/*!
 * \brief Interpolates a given box element solution to a given global position.
 *        Uses the finite element cache of the grid geometry.
 *
 * \return the interpolated Primary Variables
 * \param element The element
 * \param geometry The element geometry
 * \param fvGridGeometry The finite volume grid geometry
 * \param elemSol The primary variables at the dofs of the element
 * \param globalPos The global position
 */
template< class Element, class FVGridGeometry, class TypeTag >
typename BoxElementSolution<TypeTag>::PrimaryVariables
evalSolution(const Element& element,
             const typename Element::Geometry& geometry,
             const FVGridGeometry& fvGridGeometry,
             const BoxElementSolution<TypeTag>& elemSol,
             const typename Element::Geometry::GlobalCoordinate& globalPos)
{
    using PrimaryVariables = typename BoxElementSolution<TypeTag>::PrimaryVariables;
    using Scalar = typename PrimaryVariables::value_type;

    // interpolate the solution
    const auto& localBasis = fvGridGeometry.feCache().get(geometry.type()).localBasis();

    // evaluate the shape functions at the scv center
    const auto localPos = geometry.local(globalPos);
    std::vector< Dune::FieldVector<Scalar, 1> > shapeValues;
    localBasis.evaluateFunction(localPos, shapeValues);

    PrimaryVariables result(0.0);
    for (int i = 0; i < geometry.corners(); ++i)
    {
        auto value = elemSol[i];
        value *= shapeValues[i][0];
        result += value;
    }

    return result;
}

/*!
 * \brief Interpolates a given box element solution to a given global position.
 *        Overload of the above evalSolution() function without a given fvGridGeometry.
 *        The local basis is computed on the fly.
 *
 * \return the interpolated Primary Variables
 * \param element The element
 * \param geometry The element geometry
 * \param elemSol The primary variables at the dofs of the element
 * \param globalPos The global position
 */
template< class Element, class TypeTag >
typename BoxElementSolution<TypeTag>::PrimaryVariables
evalSolution(const Element& element,
             const typename Element::Geometry& geometry,
             const BoxElementSolution<TypeTag>& elemSol,
             const typename Element::Geometry::GlobalCoordinate& globalPos)
{
    //! The box scheme always uses linear Ansatz functions
    using FeCache = Dune::PQkLocalFiniteElementCache<CoordScalar, Scalar, dim, 1>;
    using PrimaryVariables = typename BoxElementSolution<TypeTag>::PrimaryVariables;
    using Scalar = typename PrimaryVariables::value_type;

    // obtain local finite element basis
    FeCache feCache;
    const auto& localBasis = feCache.get(geometry.type()).localBasis;

    // evaluate the shape functions at the scv center
    const auto localPos = geometry.local(globalPos);
    std::vector< Dune::FieldVector<Scalar, 1> > shapeValues;
    localBasis.evaluateFunction(localPos, shapeValues);

    PrimaryVariables result(0.0);
    for (int i = 0; i < geometry.corners(); ++i)
    {
        auto value = elemSol[i];
        value *= shapeValues[i][0];
        result += value;
    }

    return result;
}

/*!
 * \brief Interpolates a given CCElementSolution to a given global position.
 *
 * \return the (constant over the element) Primary Variables
 * \param element The element
 * \param geometry The element geometry
 * \param fvGridGeometry The finite volume grid geometry
 * \param elemSol The primary variables at the dofs of the element
 * \param globalPos The global position
 */
template< class Element, class FVGridGeometry, class TypeTag >
typename CCElementSolution<TypeTag>::PrimaryVariables
evalSolution(const Element& element,
             const typename Element::Geometry& geometry,
             const FVGridGeometry& fvGridGeometry,
             const CCElementSolution<TypeTag>& elemSol,
             const typename Element::Geometry::GlobalCoordinate& globalPos)
{
    return elemSol[0];
}

/*!
 * \brief Interpolates a given CCElementSolution to a given global position.
 *        Overload of the above evalSolution() function without a given fvGridGeometry.
 *        For compatibility reasons with the box scheme.
 *
 * \return the (constant over the element) Primary Variables
 * \param element The element
 * \param geometry The element geometry
 * \param elemSol The primary variables at the dofs of the element
 * \param globalPos The global position
 */
template< class Element, class TypeTag >
typename CCElementSolution<TypeTag>::PrimaryVariables
evalSolution(const Element& element,
            const typename Element::Geometry& geometry,
            const CCElementSolution<TypeTag>& elemSol,
            const typename Element::Geometry::GlobalCoordinate& globalPos)
{
    return elemSol[0];
}

} // namespace Dumux

#endif
