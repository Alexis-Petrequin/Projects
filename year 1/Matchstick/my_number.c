/*
** EPITECH PROJECT, 2019
** number
** File description:
** number.c
*/

#include "include.h"

int number(char *str)
{
    int i = 0;
    int nbr = 0;
    int neg = 1;

    while (str[i] != '\0') {
        while (str[i] == '-') {
            neg *= -1;
            ++i;
        }
        if (str[i] < '0' || str[i] > '9') {
            (nbr *= neg);
            return (nbr);
        }
        nbr += (str[i] - 48);
        if (str[i + 1] != '\0' && (str[i + 1] >= '0' && str[i + 1] <= '9'))
            nbr *= 10;
        ++i;
    }
    nbr *= neg;
    return (nbr);
}
