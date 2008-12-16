#ifndef MATRIXPROPERTIES
#define MATRIXPROPERTIES

#include <dumux/material/property_baseclasses.hh>

namespace Dune
{

template<class G, class RT>
class HeterogeneousSoil: public Matrix2p<G, RT>
{
public:
typedef	typename G::Traits::template Codim<0>::Entity Entity;
	typedef typename G::ctype DT;
	typedef typename Matrix2p<G, RT>::modelFlag modelFlag;

	enum
	{	n=G::dimension, m=1};

	virtual const FieldMatrix<DT,n,n> &K (const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi)
	{
		return K_;
	}
	virtual double porosity(const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi) const
	{
		return 0.2;
	}

	virtual double Sr_w(const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi, const double T = 283.15) const
	{
		return 0.2;
	}

	virtual double Sr_n(const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi, const double T = 283.15) const
	{
		return 0.2;
	}

	/* ATTENTION: define heat capacity per cubic meter! Be sure, that it corresponds to porosity!
	 * Best thing will be to define heatCap = (specific heatCapacity of material) * density * porosity*/
	virtual double heatCap(const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi) const
	{
		return 790 /* spec. heat cap. of granite */
		* 2700 /* density of granite */
		* porosity(x, e, xi);
	}

	virtual double heatCond(const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi, const double sat) const
	{
		static const double lWater = 0.6;
		static const double lGranite = 2.8;
		double poro = porosity(x, e, xi);
		double lsat = pow(lGranite, (1-poro)) * pow(lWater, poro);
		double ldry = pow(lGranite, (1-poro));
		return ldry + sqrt(sat) * (ldry - lsat);
	}

	virtual std::vector<double> paramRelPerm(const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi, const double T = 283.15) const
	{

		std::vector<double> param(2);
		if (x[0]<=300)
		{
			//linear parameters
			param[0] = 0.2;
			param[1] = 0.;
		}
		else
		{
			//Brooks-Corey parameters
			param[0] = 3; // lambda
			param[1] = 0.; // entry-pressure
		}
		return param;
	}

	virtual modelFlag relPermFlag(const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi) const
	{
//		if (x[0]<=300)
//			return 1;
//		else
			return Matrix2p<G, RT>::brooks_corey;
	}

	HeterogeneousSoil()
        :Matrix2p<G,RT>(),K_(1e-7)//,permeability(g, name, create)
	{}

	~HeterogeneousSoil()
	{}

private:
    FieldMatrix<DT,n,n> K_;
    
public:

//	RandomPermeability<G> permeability;
};

} // end namespace
#endif /*MATRIXPROPERTIES*/
