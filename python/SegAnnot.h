int SegAnnotBases(
    const double * , const int *, //signal
    const int *, const int *,  //annotations
    const int, const int, //sizes
    int *, int*, double*, //segments
    int*, int*, int*); //breakpoints

#define ERROR_BASES_NOT_INCREASING 1
#define ERROR_REGIONS_NOT_INCREASING 2
#define ERROR_LAST_BEFORE_FIRST 3
