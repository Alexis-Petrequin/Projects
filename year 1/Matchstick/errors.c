/*
** EPITECH PROJECT, 2019
** my_putstr
** File description:
** my_putstr.c
*/

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "include.h"

int letter_error(char **av, int i)
{
    for (int j = 0; av[i][j]; ++j)
        if (av[i][j] < '1' || av[i][j] > '9')
            return (84);
    return (10);
}

int starterror(int ac, char **av)
{
    int test = 0;
    if (ac != 3)
        return (84);
    for (int i = 1; i < 3; ++i){
        test = letter_error(av, i);
        if (test == 84)
            return (84);
    }
    if (number(av[1]) > 100)
        return (84);
    return (10);
}

int stickline_error(char *line, int z)
{
    for (int i = 0; line[i + 1] != '\0'; ++i) {
        if ((line[i] < '0' || line [i] > '9') && line[i] != '\0') {
            my_putstr("Error: invalid input (positive number expected)\n");
            return (101);
        }
    }
    if (number(line) > z || number(line) == 0) {
        my_putstr("Error: this line is out of range\n");
        return (101);
    }
    return (10);
}

int sticknbr_error(char *nbr, int maxsticks)
{
    for (int i = 0; nbr[i + 1] != '\0'; ++i)
        if (nbr[i] < '0' || nbr[i] > '9') {
            my_putstr("Error: invalid input (positive number expected)\n");
            return (101);
        }
    if (number(nbr) > maxsticks) {
        my_putstr("Error: you cannot remove more than ");
        my_put_nbr(maxsticks);
        my_putstr(" matches per turn\n");
        return (101);
    }
    if (number(nbr) == 0) {
        my_putstr("Error: you have to remove at least one match\n");
        return (101);
    }
    return (10);
}
