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
 * \ingroup CCMpfaDiscretization
 * \brief Base class for interaction volumes of mpfa methods.
 */
#ifndef DUMUX_DISCRETIZATION_CC_MPFA_INTERACTIONVOLUMEBASE_HH
#define DUMUX_DISCRETIZATION_CC_MPFA_INTERACTIONVOLUMEBASE_HH

#include <vector>

#include <dune/common/exceptions.hh>

namespace Dumux
{

/*
 * \ingroup CCMpfaDiscretization
 * \brief Type Traits to retrieve types associated with an implementation of a Dumux::CCMpfaInteractionVolume.
 *        You have to provide a traits class for every implementation of Dumux::CCMpfaInteractionVolume. Also,
 *        make sure to publicly export the traits class type in your interaction volume implementation.
 *        The traits should contain the following type definitions:
 *
 * \code
 * //! export the type of grid view
 * using GridView = ...;
 * //! export the type used for local indices
 * using IndexSet = ...;
 * //! export the type of interaction-volume local scvs
 * using LocalScvType = ...;
 * //! export the type of interaction-volume local scvfs
 * using LocalScvfType = ...;
 * //! export the type of used for the iv-local face data
 * using LocalFaceData = ...;
 * //! export the matrix/vector type traits to be used by the iv
 * using MatVecTraits = ...;
 * \endcode
 */

/*!
 * \ingroup CCMpfaDiscretization
 * \brief Base class for the interaction volumes of mpfa methods. It defines
 *        the interface and actual implementations should derive from this class.
 *
 * \tparam Impl The actual implementation of the interaction volume
 * \tparam T The traits class to be used
 */
template< class Impl, class T >
class CCMpfaInteractionVolumeBase
{
    // Curiously recurring template pattern
    Impl& asImp() { return static_cast<Impl&>(*this); }
    const Impl& asImp() const { return static_cast<const Impl&>(*this); }

    using GridView = typename T::GridView;
    using Element = typename GridView::template Codim<0>::Entity;

    using NodalStencilType = typename T::IndexSet::NodalGridStencilType;
    using LocalIndexType = typename T::IndexSet::LocalIndexType;
    using LocalScvType = typename T::LocalScvType;
    using LocalScvfType = typename T::LocalScvfType;

public:
    //! state the traits type publicly
    using Traits = T;

    //! Prepares everything for the assembly
    template< class Problem, class FVElementGeometry >
    void setUpLocalScope(const typename Traits::IndexSet& indexSet,
                         const Problem& problem,
                         const FVElementGeometry& fvGeometry)
    { asImp().setUpLocalScope(); }

    //! returns the number of "primary" scvfs of this interaction volume
    std::size_t numFaces() const
    { return asImp().numFaces(); }

    //! returns the number of intermediate unknowns within this interaction volume
    std::size_t numUnknowns() const
    { return asImp().numUnknowns(); }

    //! returns the number of (in this context) known solution values within this interaction volume
    std::size_t numKnowns() const
    { return asImp().numKnowns(); }

    //! returns the number of scvs embedded in this interaction volume
    std::size_t numScvs() const
    { return asImp().numScvs(); }

    //! Returns a reference to the container with the local face data. The actual type of
    //! the container depends on the interaction volume implementation. At this point we throw
    //! an exception and force the implementation to overload this function.
    const std::vector<typename Traits::LocalFaceData>& localFaceData() const
    { DUNE_THROW(Dune::NotImplemented, "Interaction volume implementation does not provide a localFaceData() funtion"); }

    //! returns the cell-stencil of this interaction volume
    const NodalStencilType& stencil() const
    { return asImp().stencil(); }

    //! returns the local scvf entity corresponding to a given iv-local scvf idx
    const LocalScvfType& localScvf(LocalIndexType ivLocalScvfIdx) const
    { return asImp().localScvf(ivLocalScvfIdx); }

    //! returns the local scv entity corresponding to a given iv-local scv idx
    const LocalScvType& localScv(LocalIndexType ivLocalScvIdx) const
    { return asImp().localScv(ivLocalScvIdx); }

    //! returns the element in which the scv with the given local idx is embedded in
    const Element& element(LocalIndexType ivLocalScvIdx) const
    { return asImp().element(); }

    //! returns the number of interaction volumes living around a vertex
    template< class NodalIndexSet >
    static std::size_t numIVAtVertex(const NodalIndexSet& nodalIndexSet)
    { return Impl::numIVAtVertex(nodalIndexSet); }

    //! adds the iv index sets living around a vertex to a given container
    //! and stores the the corresponding index in a map for each scvf
    template< class IvIndexSetContainer,
              class ScvfIndexMap,
              class NodalIndexSet,
              class FlipScvfIndexSet >
    static void addIVIndexSets(IvIndexSetContainer& ivIndexSetContainer,
                               ScvfIndexMap& scvfIndexMap,
                               const NodalIndexSet& nodalIndexSet,
                               const FlipScvfIndexSet& flipScvfIndexSet)
    { Impl::addIVIndexSets(ivIndexSetContainer, scvfIndexMap, nodalIndexSet, flipScvfIndexSet); }
};

} // end namespace Dumux

#endif