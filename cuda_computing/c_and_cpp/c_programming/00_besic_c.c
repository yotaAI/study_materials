#include<stdio.h>

//00 : If-else Condition in C
void if_condition(){
	int value=40;
	if(value==40){
		printf("The Value is %d\n",value);
	}
	else if (value>40){
		printf("The Value is Greater than %d\n",value);
	}
	else{
		printf("The Value is Less Than %d\n",value);
	}
}

//02 : Switch-Case Statement
void switch_case(){
	int value = 60;
	switch(value){
		case 60:
			printf("The value is 60\n");
			break;
		case 50:
			printf("The value is 50\n");
			break;
		default:
			printf("I don't know what the value is");
	}
}

// 03 : For Loop
void for_loop(){
	for(int i=0;i<10;i++){
	printf("%d ",i);
	}
	printf("\n");
}

// 04 : While Loop
void while_loop(){
	int i=0;
	while i<10{
		printf("%d ",i);
		i++;'
		// To Break the loop we can use break;
		if(i==5) break;

		// To Skip/ Continue we can use continue;
		if (i==7) continue;
	}
	printf("\n");
}


int main(){
	//if_condition();
	switch_case();
	return 0;
}
