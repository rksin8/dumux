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
 * \ingroup Discretization
 * \brief Properties for all models using cell-centered finite volume scheme with WMPFA
 * \note Inherit from these properties to use a cell-centered finite volume scheme with WMPFA
 */

#ifndef DUMUX_DISCRETIZATION_CC_WMPFA_HH
#define DUMUX_DISCRETIZATION_CC_WMPFA_HH

#include <dumux/common/properties.hh>
#include <dumux/common/boundaryflag.hh>
#include <dumux/common/typetraits/problem.hh>

#include <dumux/assembly/cclocalresidual.hh>

#include <dumux/discretization/method.hh>
#include <dumux/discretization/fvproperties.hh>

#include <dumux/discretization/cellcentered/subcontrolvolume.hh>
#include <dumux/discretization/cellcentered/elementboundarytypes.hh>
#include <dumux/discretization/cellcentered/wmpfa/fvgridgeometry.hh>
#include <dumux/discretization/cellcentered/wmpfa/gridvolumevariables.hh>
#include <dumux/discretization/cellcentered/wmpfa/gridfluxvariablescache.hh>
#include <dumux/discretization/cellcentered/wmpfa/subcontrolvolumeface.hh>
#include <dumux/discretization/cellcentered/wmpfa/methods.hh>

#include <dumux/discretization/cellcentered/wmpfa/interpolationoperator.hh>
#include <dumux/discretization/cellcentered/wmpfa/facedatahandle.hh>

namespace Dumux {
namespace Properties {

//! Type tag for the cell-centered wmpfa scheme.
// Create new type tags
namespace TTag {
struct CCWMpfaModel { using InheritsFrom = std::tuple<FiniteVolumeModel>; };
} // end namespace TTag

//! Set the default for the grid geometry
template<class TypeTag>
struct GridGeometry<TypeTag, TTag::CCWMpfaModel>
{
private:
    static constexpr bool enableCache = getPropValue<TypeTag, Properties::EnableGridGeometryCache>();
    using GridView = typename GetPropType<TypeTag, Properties::Grid>::LeafGridView;
public:
    using type = CCWMpfaFVGridGeometry<GridView, enableCache>;
};

//! The grid volume variables vector class
template<class TypeTag>
struct GridVolumeVariables<TypeTag, TTag::CCWMpfaModel>
{
private:
    static constexpr bool enableCache = getPropValue<TypeTag, Properties::EnableGridVolumeVariablesCache>();
    using Problem = GetPropType<TypeTag, Properties::Problem>;
    using VolumeVariables = GetPropType<TypeTag, Properties::VolumeVariables>;
public:
    using type = CCWMpfaGridVolumeVariables<Problem, VolumeVariables, enableCache>;
};

//! The grid flux variables cache vector class
template<class TypeTag>
struct GridFluxVariablesCache<TypeTag, TTag::CCWMpfaModel>
{
private:
    static constexpr bool enableCache = getPropValue<TypeTag, Properties::EnableGridFluxVariablesCache>();
    using Problem = GetPropType<TypeTag, Properties::Problem>;
    using FluxVariablesCache = GetPropType<TypeTag, Properties::FluxVariablesCache>;
    using FluxVariablesCacheFiller = GetPropType<TypeTag, Properties::FluxVariablesCacheFiller>;

    using PhysicsTraits = DataHandlePhysicsTraits<GetPropType<TypeTag, Properties::ModelTraits>>;

    struct InterpolationTraits
    {
        using GridGeometry = GetPropType<TypeTag, Properties::GridGeometry>;
        using GridView = typename GridGeometry::GridView;
        using Element = typename GridView::template Codim<0>::Entity;
        using GlobalPosition = typename Element::Geometry::GlobalCoordinate;
    };

    using InterpolationOperator = HapInterpolationOperator<InterpolationTraits, PhysicsTraits>;
    using DataHandle = FaceDataHandle<InterpolationOperator, PhysicsTraits>;

    using Traits = CCWMpfaDefaultGridFluxVariablesCacheTraits<Problem,
                                                              FluxVariablesCache, FluxVariablesCacheFiller,
                                                              DataHandle, InterpolationOperator>;

public:
    using type = CCWMpfaGridFluxVariablesCache<Traits, enableCache>;
};

//! Set the default for the ElementBoundaryTypes
template<class TypeTag>
struct ElementBoundaryTypes<TypeTag, TTag::CCWMpfaModel> { using type = CCElementBoundaryTypes; };

//! Set the BaseLocalResidual to CCLocalResidual
template<class TypeTag>
struct BaseLocalResidual<TypeTag, TTag::CCWMpfaModel> { using type = CCLocalResidual<TypeTag>; };

template<class TypeTag>
struct DiscretizationSubmethod<TypeTag, TTag::CCWMpfaModel> { static constexpr WMpfaMethod value = WMpfaMethod::avgmpfa; };
} // namespace Properties

namespace Impl {

template<class Problem>
struct ProblemTraits<Problem, DiscretizationMethod::ccwmpfa>
{
private:
    using GG = std::decay_t<decltype(std::declval<Problem>().gridGeometry())>;
    using Element = typename GG::GridView::template Codim<0>::Entity;
    using SubControlVolumeFace = typename GG::SubControlVolumeFace;
public:
    using GridGeometry = GG;
    // BoundaryTypes is whatever the problem returns from boundaryTypes(element, scvf)
    using BoundaryTypes = std::decay_t<decltype(std::declval<Problem>().boundaryTypes(std::declval<Element>(), std::declval<SubControlVolumeFace>()))>;
};

} // end namespace Impl

} // namespace Dumux

#endif
