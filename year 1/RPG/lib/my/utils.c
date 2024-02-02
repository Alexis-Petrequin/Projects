/*
** EPITECH PROJECT, 2020
** utils
** File description:
** about utils function
*/

#include "../../include/mylist.h"

char *numtostr(int nb, char *str)
{
    if (nb >= 100) {
        str[0] = nb / 100 % 10 + 48;
        str[1] = nb / 10 % 10 + 48;
        str[2] = nb % 10 + 48;
        str[3] = '\0';
    }
    if (nb < 100 && nb >= 10) {
        str[0] = nb / 10 % 10 + 48;
        str[1] = nb % 10 + 48;
        str[2] = '\0';
    }
    if (nb < 10 && nb > 0) {
        str[0] = nb + 48;
        str[1] = '\0';
    }
    if (nb <= 0) {
        str[0] = '0';
        str[1] = '\0';
    }
    return (str);
}

char *musicstr(float nb, char *str)
{
    int i = 0;
    int n = 1;

    while (n * 10 <= nb)
        n *= 10;
    for (; i <= 2 && n > 0 ; i++, n /=10) {
        str[i] = nb / n + 48;
        nb -= (str[i] - 48) * n;
    }
    str[i] = '\0';
    return (str);
}