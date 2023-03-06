#include <string.h>
#include <stdio.h>
#include <stdlib.h>

float bit_change(float x, int n) {
    int tmp = *reinterpret_cast<int*>(&x);
    int k = 32 - n;
    tmp >>= k;
    tmp <<= k;
    return *reinterpret_cast<float*>(&tmp);
}


int main(){
    float b, c;
    char readtxt[100];
    char writetxt[100];
    char line[1000000];
    char* p;
    int flag;
    FILE* read;
    FILE* write;
    for(int i=1; i<95; i++){
        int first_value = 1;  // flag for checking first value
	sprintf(readtxt, "conv2d%d.txt",i);
	sprintf(writetxt, "shift%d.txt",i);

        read = fopen(readtxt, "r");
        write = fopen(writetxt, "w");

        if (read == NULL) {
            printf("Error: Unable to open input file\n");
            return 1;
        }
        while (fscanf(read, "%f,", &b) != EOF) {
            c = bit_change(b, 16);
            if (first_value) {  // first value doesn't need comma
                fprintf(write, "%f", c);
                first_value = 0;  // set flag to false
            } else {
                fprintf(write, ",%f", c);
            }
        }
    fclose(read);
    fclose(write);
    }
    
    read = fopen("beta.txt", "r");
    write = fopen("shiftbeta.txt", "w");
    while (fgets(line, sizeof(line), read) != NULL ) {
	p = strtok(line, ",");
	flag = 0;
    	while(p != NULL){
	    if (flag != 0){
		fputs(",",write); 
            }
            b = atof(p);
            c = bit_change(b,16);
	    fprintf(write, "%f", c);
	    p=strtok(NULL, ",");
	    flag += 1;
    	}
        fputs("\n",write); 
    }
    fclose(read);
    fclose(write);

    read = fopen("mean.txt", "r");
    write = fopen("shiftmean.txt", "w");
    while (fgets(line, sizeof(line), read) != NULL ) {
	p = strtok(line, ",");
	flag = 0;
    	while(p != NULL){
	    if (flag != 0){
		fputs(",",write); 
	    }
            b = atof(p);
            c = bit_change(b,16);
	    fprintf(write, "%f", c);
	    p=strtok(NULL, ",");
	    flag += 1;
    	}
	fputs("\n",write); 
    }
    fclose(read);
    fclose(write);

    read = fopen("variance.txt", "r");
    write = fopen("shiftvariance.txt", "w");
    while (fgets(line, sizeof(line), read) != NULL ) {
	p = strtok(line, ",");
	flag = 0;
    	while(p != NULL){
	    if (flag != 0){
		fputs(",",write); 
		}
            b = atof(p);
            c = bit_change(b,16);
	    fprintf(write, "%f", c);
	    p=strtok(NULL, ",");
	    flag += 1;
    	}
	fputs("\n",write); 
    }
    fclose(read);
    fclose(write);

    read = fopen("Prediction_bias.txt", "r");
    write = fopen("shiftPrediction_bias.txt", "w");
    while (fgets(line, sizeof(line), read) != NULL ) {
	p = strtok(line, ",");
	flag = 0;
    	while(p != NULL){
	    if (flag != 0){
		fputs(",",write); 
	    }
            b = atof(p);
            c = bit_change(b,16);
	    fprintf(write, "%f", c);
	    p=strtok(NULL, ",");
	    flag += 1;
    	}
	fputs("\n",write); 
    }
    fclose(read);
    fclose(write);

    read = fopen("predictionKernel.txt", "r");
    write = fopen("shiftPrediction_kernel.txt", "w");
    while (fgets(line, sizeof(line), read) != NULL ) {
	p = strtok(line, ",");
	flag = 0;
    	while(p != NULL){
	    if (flag != 0){
		fputs(",",write); 
	    }
            b = atof(p);
            c = bit_change(b,16);
	    fprintf(write, "%f", c);
	    p=strtok(NULL, ",");
	    flag += 1;
    	}
	fputs("\n",write); 
    }
    fclose(read);
    fclose(write);

}


