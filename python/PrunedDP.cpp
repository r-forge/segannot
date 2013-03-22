#include "PrunedDP.h"
#include "polynome.h"

#include <stdio.h>

// ends is the result and needs to be pre-allocated size Kmax * Kmax.
int PrunedDP(double *profil, int nb, int Kmax, int *ends){
  double min = profil[0], max = profil[0];
  double *minCostBefore = new double[nb];
  double *minCostCurrent = new double[nb];
  double *cout_n = new double[Kmax];
  int *path = new int[nb*Kmax];
  double *tmp; //1
  int minPosition;
  double minCurrent;
  int i = 0;
  int i2 = 0;
  double somme = 0;
  int turn = 1;

  if(nb < 2){
    return ERROR_N_TOO_SMALL;
  }

  if(Kmax < 2){
    return ERROR_KMAX_TOO_SMALL;
  }

  if(Kmax > nb){
    return ERROR_KMAX_TOO_BIG;
  }

  /* Initialisation cost in 1 segment, min, max */
  while(i < nb)
    {
      if(profil[i] < min){
	min = profil[i];
      }
      if(profil[i] > max){
	max = profil[i];
      }
      somme = somme + profil[i];
      //minCostBefore[i] = - pow(somme, 2) / (i+1);
      minCostBefore[i] = -somme * somme / (i+1);
      path[i]=0;
      i++;
    }
  /* Save */
  cout_n[0] = minCostBefore[nb-1];


  /* Initialisation Polynome Cost */
  Liste * l1;  

  Polynome2 **stock= new Polynome2* [nb]; 

  i=0;
  while(i < nb)
    {
      stock[i]=new Polynome2();
      i++;	
    }


  /* Boucle turn 1 -> Kmax -1 */
  while( turn < Kmax)
    {
      /* Print turn / Kmax */
      /*fprintf(stderr, "%c Turn :   %d  / %d  ", c, turn, Kmax);*/
      /* initalisation */
      i= turn;
      i2= turn+ turn*nb;
      stock[i]->reset(1.0, -2*profil[i], minCostBefore[turn -1],  turn);
      stock[i]->setStatus(2);
      l1 = new Liste(max, min, stock[i]);
      /* Min */
      l1->computeMinOrMax(&minCurrent, &minPosition);
      minCostCurrent[i]=minCurrent;
      path[i2] = i;

      /* iterate */
      i++;
      i2++;
      while(i < nb)
	{
	  /* Slide 1 and Prune */
	  l1->computeRoots(minCostBefore[i-1]);
	  stock[i]->reset(0.0, 0.0, minCostBefore[i-1],  i);
	  l1->resetAllBorders(stock[i]);
	  l1->checkForDoublon();
	  l1->add(1.0, -2*profil[i], 0.0);

	  /* Compute Min */
	  l1->computeMinOrMax(&minCurrent, &minPosition);
	  minCostCurrent[i]=minCurrent;
	  path[i2] = minPosition;
		
	  /* iterate */
	  i++;	
	  i2++;
	}

      /* Save */
      cout_n[turn] = minCostCurrent[nb-1];
	  
      /* */
      tmp=minCostCurrent;
      minCostCurrent=minCostBefore;
      minCostBefore=tmp;
	
	
      //delete(l1);
      /* iterate */
      turn++;

    }
	
  /* Free All */
  /* free stock */
  i=0;
  while(i < nb)
    {
      delete(stock[i]);	
      i++;
    }
  delete(stock);  
  delete(minCostBefore);
  delete(minCostCurrent);
  delete(cout_n);

  // We could convert this R code to C, and return that...

  // retour <- function(path, i){
  //    chaine <- integer(i)
  //    chaine[i] <- ncol(path)
  //    for(j in i:2)  chaine[j-1] <- path[j, chaine[j]]
  //    return(chaine)
  // }

  int j, ends_from, ends_to, last_end, k=0, newend;
  for(i=0; i<Kmax; i++){ //model size.
    for(j=0;j<Kmax;j++){
      ends[i*Kmax+j]=0;
    //ends[k]=k;
    //k++;
    //printf("%d ",ends[i*Kmax + j]); //ends[i,j] in python.
    }
    //printf("\n");

    //need to subtract 1 since indices start with 0 in C.
    ends[i+Kmax*i] = nb-1;

    //for(j=0;j<nb;j++){ //probe number.
    //printf("%2d ",path[i*nb + j]); //path[i,j] in R.
    //}
    //printf("\n");

    for(j=i; j>0; j--){
      ends_from = i*Kmax + j;
      ends_to = i*Kmax + j-1;
      last_end = ends[ends_from];
      //need to subtract 1 since indices start with 0 in C.
      newend = path[j*nb + last_end]-1;
      //printf("i %d j %d from %2d to %2d last %2d new %2d\n",i,j,
	     //ends_from,ends_to,last_end, newend);
      ends[ends_to] = newend;
    }
  }
  // segmeanCO <- function(geno, Kmax){
  //   out        = colibriR_c(geno, Kmax)
  //   t.est      = t(sapply(1:Kmax,FUN=function(k){
  // c(retour(out$path, k),rep(0,Kmax-k))
  //}))
  //   t.est[1,1] = length(geno)
  //   res        = list(J.est  = out$cost,t.est  = t.est)
  //   return(res)
  // }


  delete(path);
  return 0;
}


