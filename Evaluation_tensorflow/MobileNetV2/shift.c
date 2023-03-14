#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#define bits 20

float bit_change(float x, int n){

	int tmp;
	memcpy(&tmp, &x, sizeof tmp);
	int k = 32 - n;
	tmp >>= k;
	tmp <<= k;
	float result;
	memcpy(&result, &tmp, sizeof result);

	return result;
}

int main(){
    float b, c;
    char readtxt[100];
    char writetxt[100];
    int first_value = 1;

    FILE* read = fopen("other/Conv1.txt", "r");
    FILE* write = fopen("other/shiftConv1.txt", "w");
    while (fscanf(read, "%f\n", &b) != EOF) {
        c = bit_change(b, bits);
        if (first_value) {  // first value doesn't need comma
            fprintf(write, "%f", c);
            first_value = 0;  // set flag to false
        } else {
            fprintf(write, "\n%f", c);
        }
    }
    fclose(read);
    fclose(write);
    first_value = 1;

    read = fopen("other/Conv_1.txt", "r");
    write = fopen("other/shiftConv_1.txt", "w");
    while (fscanf(read, "%f\n", &b) != EOF) {
        c = bit_change(b, bits);
        if (first_value) {  // first value doesn't need comma
            fprintf(write, "%f", c);
            first_value = 0;  // set flag to false
        } else {
            fprintf(write, "\n%f", c);
        }
    }
    fclose(read);
    fclose(write);
    first_value = 1;

    read = fopen("other/Conv_1_bn_beta.txt", "r");
    write = fopen("other/shiftConv_1_bn_beta.txt", "w");
    while (fscanf(read, "%f\n", &b) != EOF) {
        c = bit_change(b, bits);
        if (first_value) {  // first value doesn't need comma
            fprintf(write, "%f", c);
            first_value = 0;  // set flag to false
        } else {
            fprintf(write, "\n%f", c);
        }
    }
    fclose(read);
    fclose(write);
    first_value = 1;

    //19
    read = fopen("other/Conv_1_bn_mean.txt", "r");
    write = fopen("other/shiftConv_1_bn_mean.txt", "w");
    while (fscanf(read, "%f\n", &b) != EOF) {
        c = bit_change(b, bits);
        if (first_value) {  // first value doesn't need comma
            fprintf(write, "%f", c);
            first_value = 0;  // set flag to false
        } else {
            fprintf(write, "\n%f", c);
        }
    }
    fclose(read);
    fclose(write);
    first_value = 1;

    read = fopen("other/Conv_1_bn_var.txt", "r");
    write = fopen("other/shiftConv_1_bn_var.txt", "w");
    while (fscanf(read, "%f\n", &b) != EOF) {
        c = bit_change(b, bits);
        if (first_value) {  // first value doesn't need comma
            fprintf(write, "%f", c);
            first_value = 0;  // set flag to false
        } else {
            fprintf(write, "\n%f", c);
        }
    }
    fclose(read);
    fclose(write);
    first_value = 1;

    read = fopen("other/Conv_1_bn_gam.txt", "r");
    write = fopen("other/shiftConv_1_bn_gam.txt", "w");
    while (fscanf(read, "%f\n", &b) != EOF) {
        c = bit_change(b, bits);
        if (first_value) {  // first value doesn't need comma
            fprintf(write, "%f", c);
            first_value = 0;  // set flag to false
        } else {
            fprintf(write, "\n%f", c);
        }
    }
    fclose(read);
    fclose(write);
    first_value = 1;
//////////////////////////////////////////////////////////////////

    read = fopen("other/Logits_bias.txt", "r");
    write = fopen("other/shiftLogits_bias.txt", "w");
    while (fscanf(read, "%f\n", &b) != EOF) {
        c = bit_change(b, bits);
        if (first_value) {  // first value doesn't need comma
            fprintf(write, "%f", c);
            first_value = 0;  // set flag to false
        } else {
            fprintf(write, "\n%f", c);
        }
    }
    fclose(read);
    fclose(write);
    first_value = 1;

    read = fopen("other/Logits_kernel.txt", "r");
    write = fopen("other/shiftLogits_kernel.txt", "w");
    while (fscanf(read, "%f\n", &b) != EOF) {
        c = bit_change(b, bits);
        if (first_value) {  // first value doesn't need comma
            fprintf(write, "%f", c);
            first_value = 0;  // set flag to false
        } else {
            fprintf(write, "\n%f", c);
        }
    }
    fclose(read);
    fclose(write);
    first_value = 1;

////////////////////////////////////////////////////////////
    read = fopen("other/bn0_conv_0_bn_depthwise_beta.txt", "r");
    write = fopen("other/shiftbn0_conv_0_bn_depthwise_beta.txt", "w");
    while (fscanf(read, "%f\n", &b) != EOF) {
        c = bit_change(b, bits);
        if (first_value) {  // first value doesn't need comma
            fprintf(write, "%f", c);
            first_value = 0;  // set flag to false
        } else {
            fprintf(write, "\n%f", c);
        }
    }
    fclose(read);
    fclose(write);
    first_value = 1;

    read = fopen("other/bn0_conv_0_bn_depthwise_gamma.txt", "r");
    write = fopen("other/shiftbn0_conv_0_bn_depthwise_gamma.txt", "w");
    while (fscanf(read, "%f\n", &b) != EOF) {
        c = bit_change(b, bits);
        if (first_value) {  // first value doesn't need comma
            fprintf(write, "%f", c);
            first_value = 0;  // set flag to false
        } else {
            fprintf(write, "\n%f", c);
        }
    }
    fclose(read);
    fclose(write);
    first_value = 1;

    read = fopen("other/bn0_conv_0_bn_depthwise_mean.txt", "r");
    write = fopen("other/shiftbn0_conv_0_bn_depthwise_mean.txt", "w");
    while (fscanf(read, "%f\n", &b) != EOF) {
        c = bit_change(b, bits);
        if (first_value) {  // first value doesn't need comma
            fprintf(write, "%f", c);
            first_value = 0;  // set flag to false
        } else {
            fprintf(write, "\n%f", c);
        }
    }
    fclose(read);
    fclose(write);
    first_value = 1;

    read = fopen("other/bn0_conv_0_bn_depthwise_variance.txt", "r");
    write = fopen("other/shiftbn0_conv_0_bn_depthwise_variance.txt", "w");
    while (fscanf(read, "%f\n", &b) != EOF) {
        c = bit_change(b, bits);
        if (first_value) {  // first value doesn't need comma
            fprintf(write, "%f", c);
            first_value = 0;  // set flag to false
        } else {
            fprintf(write, "\n%f", c);
        }
    }
    fclose(read);
    fclose(write);
    first_value = 1;

///////////////////////////////////////////////////////////////////

    read = fopen("other/bn0_conv_0_bn_project_beta.txt", "r");
    write = fopen("other/shiftbn0_conv_0_bn_project_beta.txt", "w");
    while (fscanf(read, "%f\n", &b) != EOF) {
        c = bit_change(b, bits);
        if (first_value) {  // first value doesn't need comma
            fprintf(write, "%f", c);
            first_value = 0;  // set flag to false
        } else {
            fprintf(write, "\n%f", c);
        }
    }
    fclose(read);
    fclose(write);
    first_value = 1;

    read = fopen("other/bn0_conv_0_bn_project_gamma.txt", "r");
    write = fopen("other/shiftbn0_conv_0_bn_project_gamma.txt", "w");
    while (fscanf(read, "%f\n", &b) != EOF) {
        c = bit_change(b, bits);
        if (first_value) {  // first value doesn't need comma
            fprintf(write, "%f", c);
            first_value = 0;  // set flag to false
        } else {
            fprintf(write, "\n%f", c);
        }
    }
    fclose(read);
    fclose(write);
    first_value = 1;

   //8
    read = fopen("other/bn0_conv_0_bn_project_mean.txt", "r");
    write = fopen("other/shiftbn0_conv_0_bn_project_mean.txt", "w");
    while (fscanf(read, "%f\n", &b) != EOF) {
        c = bit_change(b, bits);
        if (first_value) {  // first value doesn't need comma
            fprintf(write, "%f", c);
            first_value = 0;  // set flag to false
        } else {
            fprintf(write, "\n%f", c);
        }
    }
    fclose(read);
    fclose(write);
    first_value = 1;

    read = fopen("other/bn0_conv_0_bn_project_variance.txt", "r");
    write = fopen("other/shiftbn0_conv_0_bn_project_variance.txt", "w");
    while (fscanf(read, "%f\n", &b) != EOF) {
        c = bit_change(b, bits);
        if (first_value) {  // first value doesn't need comma
            fprintf(write, "%f", c);
            first_value = 0;  // set flag to false
        } else {
            fprintf(write, "\n%f", c);
        }
    }
    fclose(read);
    fclose(write);
    first_value = 1;
///////////////////////////////////////////////////////////////////////////

    read = fopen("other/Conv1_beta.txt", "r");
    write = fopen("other/shiftConv1_beta.txt", "w");
    while (fscanf(read, "%f\n", &b) != EOF) {
        c = bit_change(b, bits);
        if (first_value) {  // first value doesn't need comma
            fprintf(write, "%f", c);
            first_value = 0;  // set flag to false
        } else {
            fprintf(write, "\n%f", c);
        }
    }
    fclose(read);
    fclose(write);
    first_value = 1;

    read = fopen("other/Conv1_gamma.txt", "r");
    write = fopen("other/shiftConv1_gamma.txt", "w");
    while (fscanf(read, "%f\n", &b) != EOF) {
        c = bit_change(b, bits);
        if (first_value) {  // first value doesn't need comma
            fprintf(write, "%f", c);
            first_value = 0;  // set flag to false
        } else {
            fprintf(write, "\n%f", c);
        }
    }
    fclose(read);
    fclose(write);

    first_value = 1;
    read = fopen("other/Conv1_mean.txt", "r");
    write = fopen("other/shiftConv1_mean.txt", "w");
    while (fscanf(read, "%f\n", &b) != EOF) {
        c = bit_change(b, bits);
        if (first_value) {  // first value doesn't need comma
            fprintf(write, "%f", c);
            first_value = 0;  // set flag to false
        } else {
            fprintf(write, "\n%f", c);
        }
    }
    fclose(read);
    fclose(write);

    first_value = 1;
    read = fopen("other/Conv1_variance.txt", "r");
    write = fopen("other/shiftConv1_variance.txt", "w");
    while (fscanf(read, "%f\n", &b) != EOF) {
        c = bit_change(b, bits);
        if (first_value) {  // first value doesn't need comma
            fprintf(write, "%f", c);
            first_value = 0;  // set flag to false
        } else {
            fprintf(write, "\n%f", c);
        }
    }
    fclose(read);
    fclose(write);

    first_value = 1;
    read = fopen("other/mobl0_depth.txt", "r");
    write = fopen("other/shiftmobl0_depth.txt", "w");
    while (fscanf(read, "%f\n", &b) != EOF) {
        c = bit_change(b, bits);
        if (first_value) {  // first value doesn't need comma
            fprintf(write, "%f", c);
            first_value = 0;  // set flag to false
        } else {
            fprintf(write, "\n%f", c);
        }
    }
    fclose(read);
    fclose(write);

    first_value = 1;
    read = fopen("other/mobl0_project.txt", "r");
    write = fopen("other/shiftmobl0_project.txt", "w");
    while (fscanf(read, "%f\n", &b) != EOF) {
        c = bit_change(b, bits);
        if (first_value) {  // first value doesn't need comma
            fprintf(write, "%f", c);
            first_value = 0;  // set flag to false
        } else {
            fprintf(write, "\n%f", c);
        }
    }
    fclose(read);
    fclose(write);
    first_value = 1;


    for(int i=1; i<=16; i++){
	char rname[100], wname[100];

	sprintf(rname, "bn%d/npbn_beta_depth.txt", i);
	sprintf(wname, "bn%d/shiftnpbn_beta_depth.txt", i);
	read = fopen(rname, "r");
	write = fopen(wname, "w");
	
	while (fscanf(read, "%f\n", &b) != EOF) {
            c = bit_change(b, bits);
            if (first_value) {  // first value doesn't need comma
                fprintf(write, "%f", c);
                first_value = 0;  // set flag to false
            } else {
                fprintf(write, "\n%f", c);
            }
        }
        fclose(read);
        fclose(write);
        first_value = 1;


	sprintf(rname, "bn%d/npbn_beta_expand.txt", i);
	sprintf(wname, "bn%d/shiftnpbn_beta_expand.txt", i);
	read = fopen(rname, "r");
	write = fopen(wname, "w");
	
	while (fscanf(read, "%f\n", &b) != EOF) {
            c = bit_change(b, bits);
            if (first_value) {  // first value doesn't need comma
                fprintf(write, "%f", c);
                first_value = 0;  // set flag to false
            } else {
                fprintf(write, "\n%f", c);
            }
        }
        fclose(read);
        fclose(write);
        first_value = 1;


	sprintf(rname, "bn%d/npbn_beta_project.txt", i);
	sprintf(wname, "bn%d/shiftnpbn_beta_project.txt", i);
	read = fopen(rname, "r");
	write = fopen(wname, "w");
	
	while (fscanf(read, "%f\n", &b) != EOF) {
            c = bit_change(b, bits);
            if (first_value) {  // first value doesn't need comma
                fprintf(write, "%f", c);
                first_value = 0;  // set flag to false
            } else {
                fprintf(write, "\n%f", c);
            }
        }
        fclose(read);
        fclose(write);
        first_value = 1;


	sprintf(rname, "bn%d/npbn_gam_depth.txt", i);
	sprintf(wname, "bn%d/shiftnpbn_gam_depth.txt", i);
	read = fopen(rname, "r");
	write = fopen(wname, "w");
	
	while (fscanf(read, "%f\n", &b) != EOF) {
            c = bit_change(b, bits);
            if (first_value) {  // first value doesn't need comma
                fprintf(write, "%f", c);
                first_value = 0;  // set flag to false
            } else {
                fprintf(write, "\n%f", c);
            }
        }
        fclose(read);
        fclose(write);
        first_value = 1;


	sprintf(rname, "bn%d/npbn_gam_expand.txt", i);
	sprintf(wname, "bn%d/shiftnpbn_gam_expand.txt", i);
	read = fopen(rname, "r");
	write = fopen(wname, "w");
	
	while (fscanf(read, "%f\n", &b) != EOF) {
            c = bit_change(b, bits);
            if (first_value) {  // first value doesn't need comma
                fprintf(write, "%f", c);
                first_value = 0;  // set flag to false
            } else {
                fprintf(write, "\n%f", c);
            }
        }
        fclose(read);
        fclose(write);
        first_value = 1;


	sprintf(rname, "bn%d/npbn_gam_project.txt", i);
	sprintf(wname, "bn%d/shiftnpbn_gam_project.txt", i);
	read = fopen(rname, "r");
	write = fopen(wname, "w");
	
	while (fscanf(read, "%f\n", &b) != EOF) {
            c = bit_change(b, bits);
            if (first_value) {  // first value doesn't need comma
                fprintf(write, "%f", c);
                first_value = 0;  // set flag to false
            } else {
                fprintf(write, "\n%f", c);
            }
        }
        fclose(read);
        fclose(write);
        first_value = 1;


	sprintf(rname, "bn%d/npbn_mean_depth.txt", i);
	sprintf(wname, "bn%d/shiftnpbn_mean_depth.txt", i);
	read = fopen(rname, "r");
	write = fopen(wname, "w");
	
	while (fscanf(read, "%f\n", &b) != EOF) {
            c = bit_change(b, bits);
            if (first_value) {  // first value doesn't need comma
                fprintf(write, "%f", c);
                first_value = 0;  // set flag to false
            } else {
                fprintf(write, "\n%f", c);
            }
        }
        fclose(read);
        fclose(write);
        first_value = 1;


	sprintf(rname, "bn%d/npbn_mean_expand.txt", i);
	sprintf(wname, "bn%d/shiftnpbn_mean_expand.txt", i);
	read = fopen(rname, "r");
	write = fopen(wname, "w");
	
	while (fscanf(read, "%f\n", &b) != EOF) {
            c = bit_change(b, bits);
            if (first_value) {  // first value doesn't need comma
                fprintf(write, "%f", c);
                first_value = 0;  // set flag to false
            } else {
                fprintf(write, "\n%f", c);
            }
        }
        fclose(read);
        fclose(write);
        first_value = 1;


	sprintf(rname, "bn%d/npbn_mean_project.txt", i);
	sprintf(wname, "bn%d/shiftnpbn_mean_project.txt", i);
	read = fopen(rname, "r");
	write = fopen(wname, "w");
	
	while (fscanf(read, "%f\n", &b) != EOF) {
            c = bit_change(b, bits);
            if (first_value) {  // first value doesn't need comma
                fprintf(write, "%f", c);
                first_value = 0;  // set flag to false
            } else {
                fprintf(write, "\n%f", c);
            }
        }
        fclose(read);
        fclose(write);
        first_value = 1;


	sprintf(rname, "bn%d/npbn_var_expand.txt", i);
	sprintf(wname, "bn%d/shiftnpbn_var_expand.txt", i);
	read = fopen(rname, "r");
	write = fopen(wname, "w");
	
	while (fscanf(read, "%f\n", &b) != EOF) {
            c = bit_change(b, bits);
            if (first_value) {  // first value doesn't need comma
                fprintf(write, "%f", c);
                first_value = 0;  // set flag to false
            } else {
                fprintf(write, "\n%f", c);
            }
        }
        fclose(read);
        fclose(write);
        first_value = 1;

	sprintf(rname, "bn%d/npbn_var_depth.txt", i);
	sprintf(wname, "bn%d/shiftnpbn_var_depth.txt", i);
	read = fopen(rname, "r");
	write = fopen(wname, "w");
	
	while (fscanf(read, "%f\n", &b) != EOF) {
            c = bit_change(b, bits);
            if (first_value) {  // first value doesn't need comma
                fprintf(write, "%f", c);
                first_value = 0;  // set flag to false
            } else {
                fprintf(write, "\n%f", c);
            }
        }
        fclose(read);
        fclose(write);
        first_value = 1;

	sprintf(rname, "bn%d/npbn_var_project.txt", i);
	sprintf(wname, "bn%d/shiftnpbn_var_project.txt", i);
	read = fopen(rname, "r");
	write = fopen(wname, "w");
	
	while (fscanf(read, "%f\n", &b) != EOF) {
            c = bit_change(b, bits);
            if (first_value) {  // first value doesn't need comma
                fprintf(write, "%f", c);
                first_value = 0;  // set flag to false
            } else {
                fprintf(write, "\n%f", c);
            }
        }
        fclose(read);
        fclose(write);
        first_value = 1;

///////////////////////////////////////////////////////////////////

	sprintf(rname, "mobl%d/mobl%ddepth.txt", i,i);
	sprintf(wname, "mobl%d/shiftmobl%ddepth.txt", i,i);
	read = fopen(rname, "r");
	write = fopen(wname, "w");
	
	while (fscanf(read, "%f\n", &b) != EOF) {
            c = bit_change(b, bits);
            if (first_value) {  // first value doesn't need comma
                fprintf(write, "%f", c);
                first_value = 0;  // set flag to false
            } else {
                fprintf(write, "\n%f", c);
            }
        }
        fclose(read);
        fclose(write);
        first_value = 1;

	sprintf(rname, "mobl%d/mobl%dexpand.txt", i,i);
	sprintf(wname, "mobl%d/shiftmobl%dexpand.txt", i,i);
	read = fopen(rname, "r");
	write = fopen(wname, "w");
	
	while (fscanf(read, "%f\n", &b) != EOF) {
            c = bit_change(b, bits);
            if (first_value) {  // first value doesn't need comma
                fprintf(write, "%f", c);
                first_value = 0;  // set flag to false
            } else {
                fprintf(write, "\n%f", c);
            }
        }
        fclose(read);
        fclose(write);
        first_value = 1;

	sprintf(rname, "mobl%d/mobl%dproject.txt", i,i);
	sprintf(wname, "mobl%d/shiftmobl%dproject.txt", i,i);
	read = fopen(rname, "r");
	write = fopen(wname, "w");
	
	while (fscanf(read, "%f\n", &b) != EOF) {
            c = bit_change(b, bits);
            if (first_value) {  // first value doesn't need comma
                fprintf(write, "%f", c);
                first_value = 0;  // set flag to false
            } else {
                fprintf(write, "\n%f", c);
            }
        }
        fclose(read);
        fclose(write);
        first_value = 1;
    }
}






