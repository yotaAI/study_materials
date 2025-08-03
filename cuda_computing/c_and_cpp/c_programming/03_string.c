#include<stdio.h>
#include<math.h>
#include<string.h>

void string_value();

int main(){
	string_value();
	return 1;
}

void string_value(){
	//String ==> ['H','e','l','l','o','\0'] '\0' is the null character.

	char st[100] = "Hello How are you?";
	
	printf("Enter the string");
	scanf("%s",st);
	gets(st);

	//Printf & puts
	printf("%s\n",st);
	printf("%.3s\n",st); // =>'Hel'
	printf("%5.2s\n",st); // ==> '   He'
	puts(st); // This will automatically put '\n' at the end the string.

}


void string_library(){
	char name[30]="Hello how are you?"
	int n = 0;
	// Find length of string.
	n = strlen(name); //18

	//Concat 2 string V1
	char s1[] = "Hello";
	char s2[] = "How are";
	strcat(s1,s2); // s2 will be appended to s1 and return pointer to s1.
	puts(s1);

	// Concat 2 string V2
	strncat(s1,s2,3); // => 'HelloHow'
	
}

