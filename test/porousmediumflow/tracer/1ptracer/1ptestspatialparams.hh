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
 * \ingroup TracerTests
 * \brief The spatial params the incompressible test
 */
#ifndef DUMUX_INCOMPRESSIBLE_ONEP_TEST_SPATIAL_PARAMS_HH
#define DUMUX_INCOMPRESSIBLE_ONEP_TEST_SPATIAL_PARAMS_HH

#include <random>
#include <dumux/material/spatialparams/fv1p.hh>

namespace Dumux
{
/*!
 * \ingroup TracerTests
 * \brief The spatial params the incompressible test
 */
template<class TypeTag>
class OnePTestSpatialParams : public FVSpatialParamsOneP<TypeTag>
{
    using ParentType = FVSpatialParamsOneP<TypeTag>;
    using Problem = typename GET_PROP_TYPE(TypeTag, Problem);
    using Scalar = typename GET_PROP_TYPE(TypeTag, Scalar);
    using GridView = typename GET_PROP_TYPE(TypeTag, GridView);
    using Element = typename GridView::template Codim<0>::Entity;
    using FVElementGeometry = typename GET_PROP_TYPE(TypeTag, FVGridGeometry)::LocalView;
    using SubControlVolume = typename FVElementGeometry::SubControlVolume;
    using ElementSolutionVector = typename GET_PROP_TYPE(TypeTag, ElementSolutionVector);

    static constexpr int dimWorld = GridView::dimensionworld;
    using GlobalPosition = Dune::FieldVector<Scalar, dimWorld>;

public:
    using PermeabilityType = Scalar;
    OnePTestSpatialParams(const Problem& problem)
    : ParentType(problem), K_(problem.fvGridGeometry().gridView().size(0), 0.0)
    {
        permeability_ = getParam<Scalar>("SpatialParams.Permeability");
        permeabilityLens_ = getParam<Scalar>("SpatialParams.PermeabilityLens");

        lensLowerLeft_ = getParam<GlobalPosition>("SpatialParams.LensLowerLeft");
        lensUpperRight_ =getParam<GlobalPosition>("SpatialParams.LensUpperRight");

        // generate random fields
        std::mt19937 rand(0);
        std::lognormal_distribution<Scalar> K(std::log(permeability_), std::log(permeability_)*0.1);
        std::lognormal_distribution<Scalar> KLens(std::log(permeabilityLens_), std::log(permeabilityLens_)*0.1);
        for (const auto& element : elements(problem.fvGridGeometry().gridView()))
        {
            const auto eIdx = problem.fvGridGeometry().elementMapper().index(element);
            const auto globalPos = element.geometry().center();
            K_[eIdx] = isInLens_(globalPos) ? KLens(rand) : K(rand);
        }
    }

    /*!
     * \brief Function for defining the (intrinsic) permeability \f$[m^2]\f$.
     *
     * \param element The element
     * \param scv The sub control volume
     * \param elemSol The element solution vector
     * \return the intrinsic permeability
     */
    const PermeabilityType& permeability(const Element& element,
                                         const SubControlVolume& scv,
                                         const ElementSolutionVector& elemSol) const
    {
        return K_[scv.dofIndex()];

        // if (isInLens_(scv.dofPosition()))
        //     return permeabilityLens_;
        // else
        //     return permeability_;
    }

    /*!
     * \brief Function for defining the porosity.
     *        That is possibly solution dependent.
     *
     * \param element The current element
     * \param scv The sub-control volume inside the element.
     * \param elemSol The solution at the dofs connected to the element.
     * \return the porosity
     */
    Scalar porosity(const Element &element,
                        const SubControlVolume &scv,
                        const ElementSolutionVector &elemSol) const
    { return 0.2; }

    //! Reference to the k field
    const std::vector<Scalar>& getKField() const
    { return K_; }

private:
    bool isInLens_(const GlobalPosition &globalPos) const
    {
        for (int i = 0; i < dimWorld; ++i) {
            if (globalPos[i] < lensLowerLeft_[i] + eps_ || globalPos[i] > lensUpperRight_[i] - eps_)
                return false;
        }
        return true;
    }

    GlobalPosition lensLowerLeft_;
    GlobalPosition lensUpperRight_;

    Scalar permeability_, permeabilityLens_;

    std::vector<Scalar> K_;

    static constexpr Scalar eps_ = 1.5e-7;
};

} // end namespace Dumux

#endif
