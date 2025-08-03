#include<stdio.h>
#define N 5; // Define N=5

void array();

int main(){
	array();
	return 0;
}
void array(){
	//////Initilaization	
	// Integer array
	int a[60]; // Array of 60 size. Initialized with garbage values.
	int len_array = sizeof(a)/sizeof(int);
	printf("Size of array is %d\n",len_array);
	
	// Initialize in compile time
	int b[4] = {0,1,2,3};
	int c[] = {1,2,3,4,4,5,6};
	int d[5] = {0}; //Initialize all values with 0	
	// int d[3] = {}; --> This will give error.

	// Runtime Initialization
	for(int i=0;i<len_array;i++){
		a[i]=i;
	}
		
}

void 2d_array(){
	int row_size,column_size=5,4;
	int arr[row_size][column_size];
	

}
