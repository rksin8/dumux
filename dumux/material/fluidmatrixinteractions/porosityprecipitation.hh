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
 * \brief Class for the evaluation of the porosity subject to precipitation.
 */
#ifndef DUMUX_POROSITY_PRECIPITATION_HH
#define DUMUX_POROSITY_PRECIPITATION_HH

#include <dumux/discretization/scvoperator.hh>

namespace Dumux
{

/*!
 * \ingroup fluidmatrixinteractionslaws
 */

/**
 * \brief Calculates the porosity depeding on the volume fractions of precipitated minerals.
 */
template<class TypeTag>
class PorosityPrecipitation
{
    using Scalar = typename GET_PROP_TYPE(TypeTag, Scalar);
    using Problem = typename GET_PROP_TYPE(TypeTag, Problem);
    using GridView = typename GET_PROP_TYPE(TypeTag, GridView);
    using ElementSolution = typename GET_PROP_TYPE(TypeTag, ElementSolutionVector);
    using SubControlVolume = typename GET_PROP_TYPE(TypeTag, SubControlVolume);
    using ScvOperator = SubControlVolumeOperator<TypeTag>;

    static const int dim = GridView::dimension;
    static const int dimWorld = GridView::dimensionworld;
    static const int numComponents = GET_PROP_VALUE(TypeTag, NumComponents);
    static const int numSolidPhases = GET_PROP_VALUE(TypeTag, NumSPhases);

    using Element = typename GridView::template Codim<0>:: Entity;
    using Tensor = Dune::FieldMatrix<Scalar, dimWorld, dimWorld>;

    using InitPoroField = std::function<Scalar(const Element&, const SubControlVolume&)>;

public:
    void init(InitPoroField&& initPoro,
              InitPoroField&& minPoro)
    {
        initPoro_ = initPoro;
        minPoro_ = minPoro;
    }

    // calculates the porosity in a sub-control volume
    Scalar evaluatePorosity(const Element& element,
                            const SubControlVolume& scv,
                            const ElementSolution& elemSol) const
    {
        auto priVars = ScvOperator::evaluateSolution(element, scv, elemSol);

        Scalar sumPrecipitates = 0.0;
        for (unsigned int solidPhaseIdx = 0; solidPhaseIdx < numSolidPhases; ++solidPhaseIdx)
            sumPrecipitates += priVars[numComponents + solidPhaseIdx];

        return std::max(minPoro_(element, scv), initPoro_(element, scv) - sumPrecipitates);
    }

private:
    InitPoroField initPoro_;
    InitPoroField minPoro_;
};

} // namespace Dumux

#endif
