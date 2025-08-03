#include<stdio.h>
#include<string.h>

void pointers();
void pointer_arrithmatic();
void different_pointer();
int sum(int,int);
int main(){
	different_pointer();
	return 0;
}

void pointers(){
	// &--> Address of , * --> Indicator/dereferrence.
	
	int* int_ptr; // integer pointer
	int a = 1000;
	int_ptr = &a; // '&a' means address of 'a'
	printf("%d",*int_ptr); //1000
	
	//Pointer to pointer
	int** pointer_to_pointer = &int_ptr;
	printf("A = %d",**pointer_to_pointer); // 1000

}

void pointer_arrithmatic(){
	int a = 10;
	int *p = &a;
	*p+=2;
	printf("%d\n",*p);
	printf("%d\n",a);
	
	p+=1; // This means value stored in pointer + sizeof(the type of pointer)
		// Here p+=1 means (location of a + sizeof(int))

	int arr[] = {1,2,3,4,5};
	p = &arr[0];
	for(int i=0;i<5;i++){
		printf("%d",*(p+i));
	}
}	

void different_pointer(){
	
	//Void Pointer
	void *vp; // I can type-cast this datatype.
	int a = 10;
	vp = &a;
	printf("%d",*(int*)vp);
	
	//Null Pointer
	int *pt = NULL; 
	//pt = (int*)malloc(5*sizeof(int));
	
	//Dangling pointer
	int *ptr = (int*)malloc(sizeof(int));// Dinamically assigned memory.
	*ptr= 10;
	free(ptr); // Freed the memory.
	//Now ptr is still popinting to the memory. This is called dangling pointer.
	ptr=NULL; //Moving from dangling pointer.
	
	//Function Pointer
	int (*sptr)(int,int) = &sum;
	int s = (*sptr)(2,3);
	printf("%d",s);
}

int sum(int a, int b){
	return a+b;
}










































