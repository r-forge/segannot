#include <math.h>
#include <float.h>
#include <stdlib.h>

class Polynome2 {
 private:
     double a2, a1, a0;
     /* racine du polynome - A*/
     double rac1, rac2;
     /* status of the polynome 0 = roots not computed, 1 = computed*/
     int status;
     int origine;
 public:
     /* constructors and destructors */
     Polynome2()
     : a0(0),a1(0),a2(0),rac1(0.),rac2(0.),status(0),origine(0) {}
     ~Polynome2() {}
     Polynome2(double A2, double A1, double A0, int origine_)
     : a0(A0),a1(A1),a2(A2),rac1(0.),rac2(0.),status(0),origine(origine_) {}
     /* a few operations */
	/* reset */
     inline
     void reset(double A2, double A1, double A0, int origine_);
     /* getter and setter */
     inline
     double geta2();
     inline
     double geta1();
     inline
     double geta0();
     inline
     double getRacine1();
     inline
     double getRacine2();
     inline
     void seta2(double);
     inline
     void seta1(double);
     inline
     void seta0(double);
     inline
     void setRacine1(double);
     inline
     void setRacine2(double);
     inline
     void setStatus(int);
     inline
     int getStatus();
     inline
     int getOrigine();
     /* Delta and others */
     inline
     double eval(double);
     inline
     double delta();
     /* Delta  of the Polynome - double */
     inline
     double delta(double);
     inline
     void roots();
     /* Roots  of the Polynome - double */
     inline
     void roots(double);
     inline
     void add(double, double, double);
     inline
     void minOrMax(double *, double *, int *);
     /* print and others */
     inline
     void show();
};
/* reset */
void Polynome2::reset(double A2, double A1, double A0, int origine_)
{
	a2= A2;
    a1= A1;
    a0= A0;
    rac1=0.;//*A
    rac2=0.;//*A
    status=0;
    origine= origine_;
}
/* getter and setter */
double Polynome2::geta2()
{
	return(a2);
}
double Polynome2::geta1()
{
	return(a1);
}
double Polynome2::geta0()
{
	return(a0);
}
void Polynome2::seta2(double a2_)
{
	a2=a2_;
}
void Polynome2::seta1(double a1_)
{
	a1=a1_;
}
void Polynome2::seta0(double a0_)
{
	a0=a0_;
}

void Polynome2::setRacine1(double rac1_)
{
	rac1=rac1_;
}

double Polynome2::getRacine1()
{
	return(this->rac1);
}
void Polynome2::setRacine2(double rac2_)
{
	rac2=rac2_;
}

double Polynome2::getRacine2()
{
	return(this->rac2);
}

int Polynome2::getStatus(){
	return(this->status);
}
int Polynome2::getOrigine()
{
	return(this->origine);
}
void Polynome2::setStatus(int status_)
{
	status=status_;
}
/* Delta and Others */
double Polynome2::eval(double X)
{
	return( (a2) * X*X + (a1)*X + (a0) );
}
void Polynome2::minOrMax(double *minOrMax, double *tmp, int *origine_)
{
	if(this->getStatus() != 0)
	{

		*tmp = -0.25 * a1*a1 / (a2) + (a0) ;
		if((*tmp) < (*minOrMax) )
		{
			(*minOrMax) = (*tmp);
			(*origine_) = this->getOrigine();
		}
		this->setStatus(0);
	}
}
double Polynome2::delta()
{
	return( a1*a1 - 4* (a2)* (a0));
}

double Polynome2::delta(double a0_)
{
	return( a1*a1 - 4* (a2)* (a0 - a0_));
}
void Polynome2::roots()
{
	if(this->getStatus() == 0)
	{
		double delta = this->delta();
		if(delta == 0)
		{
			this->setRacine1( -(a1)/(2*(a2)));
			this->setRacine2(0.);//*A
		}
		if(delta < 0)
		{
			this->setRacine1(0.);//*A
			this->setRacine2(0.);//*A
		}
		if(delta > 0)
		{
			delta = sqrt(delta);
			this->setRacine1( (-(a1) + delta ) / (2*(a2)));
			this->setRacine2( (-(a1) - delta ) / (2*(a2)));
		}
		this->setStatus(1);
	}
}
void Polynome2::roots(double a0_)
{
	if(this->getStatus() != 1)
	{
		double delta = this->delta(a0_);
		if(delta == 0)
		{
			this->setRacine1( -(a1)/(2*(a2)));
			this->setRacine2(0.);//*A
		}
		if(delta < 0)
		{
			this->setRacine1(0.);//*A
			this->setRacine2(0.);//*A
		}
		if(delta > 0)
		{
			delta = sqrt(delta);
			this->setRacine1( (-(a1) + delta ) / (2*(a2)));
			this->setRacine2( (-(a1) - delta ) / (2*(a2)));
		}
		this->setStatus(1);
	}
}
void Polynome2::add(double a2_, double a1_, double a0_)
{
	if( this->getStatus() != 2)
	{
		this->a2 = a2+ a2_;
		this->a1 = a1+ a1_;
		this->a0 = a0+ a0_;
		this->setStatus(2);
	}
}
/* print and others */
void Polynome2::show()
{

   //std::cout << this->geta2() << " x^2 + " << this->geta1() << " x + " << this->geta0() << std::endl;
   //std::cout << "Rc1 : " << this->getRacine1() << " , Rc2 : " << this->getRacine2() << ", St : " << this->getStatus() << ", Or : " << this->getOrigine() << std::endl;
   //std::cout << "-----------------------" <<std::endl;
}


class Liste {
private:
  double max, min;
  Polynome2 *poly;
  Liste *next;
public:
  /* constructors and destructors */
  Liste()
  : max(0.), min(0.), next(NULL), poly(NULL) {}
  Liste(double max_, double min_)
  : max(max_), min(min_), next(NULL) , poly(NULL){}
  Liste(double max_, double min_, Polynome2 *poly_)
  : max(max_), min(min_), next(NULL), poly(poly_) {}
  Liste(Polynome2 *poly_)
  : max(0.), min(0.), next(NULL), poly(poly_) {}
  ~Liste(){
    delete next;
    delete poly;
  }
  /* fonction setter and getter */
  inline
  double getMax();
  inline
  void setMax(double max_);
  inline
  double getMin();
  inline
  void setMin(double min_);
  inline
  void setPolynome(Polynome2 * poly_);
  inline
  Polynome2 *getPolynome();
  inline
  Liste * getNext();
  inline
  void setNext(Liste * next_);
  /* Useful */
  inline
  void setToNull();
  inline
  void insert(Liste * maillon_);
  inline
  int compte();
  inline
  Liste * removeDoublon();
  inline
  void checkForDoublon();
  /* show and others */
  void show();
  void showAllNext();
  /* */
  inline
  void computeRoots(double);
  inline
  void add(double, double, double);
  inline
  void computeMinOrMax(double*, int*);
  void resetMaillonBorders(Polynome2*);
  void resetAllBorders(Polynome2*);
};
/* Setter and Getter */
/* */
double Liste::getMax()
{
  return(this->max);
}
void Liste::setMax(double max_)
{
  this->max = max_;
}
double Liste::getMin()
{
  return(this->min);
}
void Liste::setMin(double min_)
{
  this->min = min_;
}
void Liste::setPolynome(Polynome2 * poly_)
{
  this->poly=poly_;
}
Polynome2* Liste::getPolynome()
{
  return(this->poly	);
}
/* */
Liste * Liste::getNext()
{
  return(this->next);
}
void Liste::setNext(Liste * next_)
{
  this->next = next_;
}
void Liste::setToNull()
{
  max=0.;
  min=0.;
  poly=NULL;
  next=NULL;
}
void Liste::insert(Liste * maillon_)
{
  maillon_->setNext(this->getNext());
  this->setNext(maillon_);
}
int Liste::compte(){
  Liste *l;
  int tmp = 0;
  l=this;
  while(l != NULL){
      tmp= tmp+1;
      l=l->getNext();
  }
  return(tmp);
}
Liste * Liste::removeDoublon()
{
  Liste *next = this->getNext();
  if(next != NULL)
    {
      if(next->getPolynome() == this->getPolynome())
        {
          //std::cerr<<"erase"<<std::endl;
          this->setMin(next->getMin());
          this->setNext(next->getNext());
          next->setToNull();
          delete next;
          return(this);
        } else
          {
            return(next);
          }
    } else
      {
        return(NULL);
      }
}
void Liste::checkForDoublon()
{
  Liste *l = this;
  while(l != NULL)
    {
      l=l->removeDoublon();
    }
}
void Liste::computeRoots(double a0_)
{
  Liste *l;
  l=this;
  while(l != NULL)
    {
      l->getPolynome()->roots(a0_);
      l=l->getNext();
    }
}
void Liste::add(double a2_, double a1_, double a0_)
{
  Liste *l;
  l=this;
  while(l != NULL)
    {
      l->getPolynome()->add(a2_, a1_, a0_);
      l=l->getNext();
    }
}
void Liste::computeMinOrMax(double * min, int * which)
{
  Liste *l;
  double tmp = INFINITY;
  *min = INFINITY;
  * which=-1;
  l=this;
  while(l != NULL)
    {
      l->getPolynome()->minOrMax(min, &tmp, which);
      l=l->getNext();
    }
}

void Liste::show()
{
        //std::cout << "Max : " << this->getMax() << ", Min : " << this->getMin()<< std::endl;
        this->poly->show();
}
void Liste::showAllNext()
{
        Liste *l;
        l = this;
        while(l != NULL)
        {
                l->show();
                l=l->getNext();
        }

}
void Liste::resetMaillonBorders(Polynome2 *poly_)
{
	//if(this->getPolynome()->getRacine2() == NAN)
	if(this->getPolynome()->getRacine2() == 0.)
	//if( isnan(this->getPolynome()->getRacine2()) )
	{
		this->setPolynome(poly_);
	} else if(this->getPolynome()->getRacine1() >= this->getMax())
	{
		if(this->getPolynome()->getRacine2() >= this->getMax())
		{
			this->setPolynome(poly_);
		} else if(this->getPolynome()->getRacine2() > this->getMin()) 
			{
				Liste * maillon= new Liste(this->getPolynome()->getRacine2(), this->getMin(), poly_);
				this->insert(maillon);
				this->setMin(this->getPolynome()->getRacine2());
			} else 
			{
			}
		
	
	} else if( this->getPolynome()->getRacine1() > this->getMin())
		{
		
			if(this->getPolynome()->getRacine2() > this->getMin())
			{
				Liste *maillon3 = new Liste(this->getPolynome()->getRacine2(), this->getMin(), poly_);
				Liste *maillon2 = new Liste(this->getPolynome()->getRacine1(), this->getPolynome()->getRacine2(), this->getPolynome());
				this->setMin(this->getPolynome()->getRacine1());
				this->setPolynome(poly_);
				this->insert(maillon3);
				this->insert(maillon2);
			} else
			{
				Liste *maillon2 = new Liste(this->getPolynome()->getRacine1(), this->getMin(), this->getPolynome());
				this->setMin(this->getPolynome()->getRacine1());
				this->setPolynome(poly_);
				this->insert(maillon2);
			}
		
		} else 
		{
			this->setPolynome(poly_);
		}
}
void Liste::resetAllBorders(Polynome2 *poly_)
{

	Liste *lCurrent, *lNext;
	lCurrent = this;
	lNext = this->getNext();
	lCurrent->resetMaillonBorders(poly_);
	lCurrent=lNext;
	while(lCurrent != NULL)
	{
		lNext=lCurrent->getNext();	
		lCurrent->resetMaillonBorders(poly_);
		lCurrent=lNext;
	}
}

