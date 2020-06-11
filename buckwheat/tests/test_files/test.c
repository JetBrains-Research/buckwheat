#include <stdio.h>
#include <math.h>
/* This function converts the octal number "octalnum" to the
 * decimal number and returns it.
 */
long octalToDecimal(int octalnum)
{
    int decimalnum = 0, temp = 0;

    while(octalnum != 0)
    {
        decimalnum = decimalnum + (octalnum%10) * pow(8,temp);
        temp++;
        octalnum = octalnum / 10;
    }

    return decimalnum;
}
int main()
{
    int octalnum;

    printf("Enter an octal number: ");
    scanf("%d", &octalnum);

    printf("Equivalent decimal number is: %ld", octalToDecimal(octalnum));

    return 0;
}
