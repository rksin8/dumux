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
 * \brief This file contains the data which is required to calculate the
 *        energy fluxes over a face of a finite volume.
 *
 * This means concentration temperature gradients and heat conductivity at
 * the integration point.
 */
#ifndef DUMUX_STOKESNCNI_FLUX_VARIABLES_HH
#define DUMUX_STOKESNCNI_FLUX_VARIABLES_HH

#include <dumux/common/math.hh>
#include <dumux/freeflow/stokesnc/fluxvariables.hh>

namespace Dumux
{

/*!
 * \ingroup BoxStokesncniModel
 * \ingroup ImplicitFluxVariables
 * \brief This template class contains data which is required to
 *        calculate the energy fluxes over a face of a finite
 *        volume for the non-isothermal compositional n-component Stokes box model.
 *
 * This means temperature gradients and heat conductivity
 * at the integration point of a SCV face or boundary face.
 */
template <class TypeTag>
class StokesncniFluxVariables : public StokesncFluxVariables<TypeTag>
{
    friend class StokesFluxVariables<TypeTag>; // be friends with grandparent
    friend class StokesncFluxVariables<TypeTag>; // be friends with parent
    typedef StokesncFluxVariables<TypeTag> ParentType;
    typedef typename GET_PROP_TYPE(TypeTag, Scalar) Scalar;
    typedef typename GET_PROP_TYPE(TypeTag, GridView) GridView;

    typedef typename GET_PROP_TYPE(TypeTag, Indices) Indices;

    typedef typename GET_PROP_TYPE(TypeTag, Problem) Problem;
    typedef typename GET_PROP_TYPE(TypeTag, ElementVolumeVariables) ElementVolumeVariables;

    enum { dim = GridView::dimension };

    //number of components
    enum {  numComponents = Indices::numComponents };

    typedef typename GridView::template Codim<0>::Entity Element;
    typedef Dune::FieldVector<Scalar, dim> DimVector;

    typedef typename GET_PROP_TYPE(TypeTag, FVElementGeometry) FVElementGeometry;

public:
    /*!
     * \brief Returns the temperature \f$\mathrm{[K]}\f$ at the integration point.
     */
    const Scalar temperature() const
    { return temperature_; }

    /*!
     * \brief Returns the thermal conductivity \f$\mathrm{[W/(m*K)]}\f$ at the integration point.
     */
    const Scalar thermalConductivity() const
    { return thermalConductivity_; }

    /*!
     * \brief Returns the specific isobaric heat capacity \f$\mathrm{[J/(kg*K)]}\f$
     *        at the integration point.
     */
    Scalar heatCapacity() const
    { return heatCapacity_; }

    /*!
     * \brief Return the enthalpy of a component \f$\mathrm{[J/kg]}\f$ at the integration point.
     */
    const Scalar componentEnthalpy(int componentIdx) const
    { return componentEnthalpy_[componentIdx]; }

    /*!
     * \brief Returns the temperature gradient \f$\mathrm{[K/m]}\f$ at the integration point.
     */
    const DimVector &temperatureGrad() const
    { return temperatureGrad_; }

    /*!
     * \brief Return the thermal eddy conductivity \f$\mathrm{[W/(m*K)]}\f$ (if implemented).
     */
    const Scalar thermalEddyConductivity() const
    { return 0; }


protected:
    void calculateValues_(const Problem &problem,
                          const Element &element,
                          const ElementVolumeVariables &elemVolVars)
    {
        ParentType::calculateValues_(problem, element, elemVolVars);

        temperature_ = Scalar(0);
        thermalConductivity_ = Scalar(0);
        heatCapacity_ = Scalar(0);
        temperatureGrad_ = Scalar(0);

        // calculate gradients and secondary variables at IPs
        for (int scvIdx = 0;
             scvIdx < this->fvGeometry_().numScv;
             scvIdx++) // loop over vertices of the element
        {
            temperature_ += elemVolVars[scvIdx].temperature() *
                this->face().shapeValue[scvIdx];

            thermalConductivity_ += elemVolVars[scvIdx].thermalConductivity() *
                this->face().shapeValue[scvIdx];

            heatCapacity_ += elemVolVars[scvIdx].heatCapacity() *
                this->face().shapeValue[scvIdx];

            // the gradient of the temperature at the IP
            DimVector grad = this->face().grad[scvIdx];
            grad *= elemVolVars[scvIdx].temperature();
            temperatureGrad_ += grad;
        }
        Valgrind::CheckDefined(temperature_);
        Valgrind::CheckDefined(thermalConductivity_);
        Valgrind::CheckDefined(heatCapacity_);
        Valgrind::CheckDefined(temperatureGrad_);

        for (unsigned int i = 0; i < numComponents; ++i)
        {
            componentEnthalpy_[i] = Scalar(0.0);
            for (int scvIdx = 0; scvIdx < this->fvGeometry_().numScv; scvIdx++) // loop over vertices of the element
            {
                componentEnthalpy_[i] += elemVolVars[scvIdx].componentEnthalpy(i)
                                         * this->face().shapeValue[scvIdx];
            }
            Valgrind::CheckDefined(componentEnthalpy_[i]);
        }
    }

    Scalar temperature_;
    Scalar thermalConductivity_;
    Scalar heatCapacity_;
    Scalar componentEnthalpy_[numComponents];
    DimVector temperatureGrad_;
};

} // end namespace

#endif
