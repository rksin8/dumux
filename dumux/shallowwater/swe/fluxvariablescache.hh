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
 * \ingroup SweModel
 * \copydoc Dumux::SweVariablesCache
 */
#ifndef DUMUX_SWE_FLUXVARIABLESCACHE_HH
#define DUMUX_SWE_FLUXVARIABLESCACHE_HH

#include <dumux/common/properties.hh>
#include <dumux/discretization/methods.hh>

namespace Dumux
{
// forward declaration
template<class TypeTag, DiscretizationMethod Method>
class SweFluxVariablesCacheImplementation
{};

/*!
 * \ingroup SweModel
 * \brief The flux variables cache classes for the SWEs model.
 *        Store flux stencils and data required for flux calculation
 */
template<class TypeTag>
using SweFluxVariablesCache = SweFluxVariablesCacheImplementation<TypeTag, GET_PROP_TYPE(TypeTag, FVGridGeometry)::discMethod>;

/*!
 * \ingroup SweModel
 * \brief The flux variables cache classes for the SWEs model.
 *        Store flux stencils and data required for flux calculation. <BR>
 */
template<class TypeTag>
class SweFluxVariablesCacheImplementation<TypeTag,DiscretizationMethod::cctpfa>
{
    using Problem = typename GET_PROP_TYPE(TypeTag, Problem);
    using GridView = typename GET_PROP_TYPE(TypeTag, GridView);
    using FluxVariables = typename GET_PROP_TYPE(TypeTag, FluxVariables);
    using FVElementGeometry = typename GET_PROP_TYPE(TypeTag, FVGridGeometry)::LocalView;
    using ElementVolumeVariables = typename GET_PROP_TYPE(TypeTag, GridVolumeVariables)::LocalView;
    using SubControlVolumeFace = typename FVElementGeometry::SubControlVolumeFace;
    using Element = typename GridView::template Codim<0>::Entity;
    using IndexType = typename GridView::IndexSet::IndexType;

public:
    //! Do nothing so far.
    void update(const Problem& problem,
                const Element& element,
                const FVElementGeometry& fvGeometry,
                const ElementVolumeVariables& elemVolVars,
                const SubControlVolumeFace &scvf)
    { //so far empty for SWEs, the porous media models compute the gradients here
    }

    //! Do nothing so far.
    void updateAdvection(const Problem& problem,
                const Element& element,
                const FVElementGeometry& fvGeometry,
                const ElementVolumeVariables& elemVolVars,
                const SubControlVolumeFace &scvf)
    {}
};

template<class TypeTag>
class SweAdvectiveFluxCacheFiller
{
    using Problem = typename GET_PROP_TYPE(TypeTag, Problem);
    using FVElementGeometry = typename GET_PROP_TYPE(TypeTag, FVGridGeometry)::LocalView;
    using SubControlVolumeFace = typename FVElementGeometry::SubControlVolumeFace;
    using Element = typename GET_PROP_TYPE(TypeTag, GridView)::template Codim<0>::Entity;
    using ElementVolumeVariables = typename GET_PROP_TYPE(TypeTag, GridVolumeVariables);
    using FluxVariablesCache = typename GET_PROP_TYPE(TypeTag, FluxVariablesCache);

public:
    //! For SWEs this is a dummy interface
    //! This interface has to be met by any advection-related cache filler class
    //! TODO: Probably get cache type out of the filler
    template<class FluxVariablesCacheFiller>
    static void fill(FluxVariablesCache& scvfFluxVarsCache,
                     const Problem& problem,
                     const Element& element,
                     const FVElementGeometry& fvGeometry,
                     const ElementVolumeVariables& elemVolVars,
                     const SubControlVolumeFace& scvf,
                     const FluxVariablesCacheFiller& fluxVarsCacheFiller)
    {}
};

template<class TypeTag>
class SweAdvectiveFluxCache
{
    using Scalar = typename GET_PROP_TYPE(TypeTag, Scalar);
    using Problem = typename GET_PROP_TYPE(TypeTag, Problem);
    using AdvectionType = typename GET_PROP_TYPE(TypeTag, AdvectionType);
    using FVElementGeometry = typename GET_PROP_TYPE(TypeTag, FVGridGeometry)::LocalView;
    using SubControlVolumeFace = typename FVElementGeometry::SubControlVolumeFace;
    using Element = typename GET_PROP_TYPE(TypeTag, GridView)::template Codim<0>::Entity;
    using ElementVolumeVariables = typename GET_PROP_TYPE(TypeTag, GridVolumeVariables)::LocalView;

public:
    //using Filler = ShallowWaterAdvectiveFluxCacheFiller<TypeTag>;

    void updateAdvection(const Problem& problem,
                         const Element& element,
                         const FVElementGeometry& fvGeometry,
                         const ElementVolumeVariables& elemVolVars,
                         const SubControlVolumeFace &scvf)
    {}
};

} // end namespace

#endif
