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

int player_line(int z)
{
    char *line = NULL;
    size_t len = 0;
    int error = 0;

    my_putstr("Line: ");
    error = getline(&line, &len, stdin);
    if (error == -1)
    return (0);
    error = stickline_error (line, z);
    if (error == 84)
        return (84);
    if (error == 101)
        return (101);
    return (number(line));
}

int player_nbr(int maxsticks)
{
    char *nbr = NULL;
    size_t len = 0;
    char **test = malloc(sizeof(char));
    int error = 0;

    my_putstr("Matches: ");
    error = getline(&nbr, &len, stdin);
    if (error == -1)
    return (0);
    error = sticknbr_error(nbr, maxsticks);
    if (error == 84)
        return (84);
    if (error == 101)
        return (101);
    return (number(nbr));
}

int stickgestion(char **map, int z, int maxsticks)
{
    static int line = 0;
    static int nbr = 0;
    int test = 0;

    if (line != 101 && nbr != 101)
        my_putstr("Your turn:\n");
    line = player_line(z);
    if (line == 0)
        return (0);
    if (line == 84)
        return (84);
    if (line == 101)
        return (101);
    nbr = player_nbr(maxsticks);
    if (nbr == 0)
        return (0);
    if (nbr == 84)
        return (84);
    if (nbr == 101)
        return (101);
    test = checkupdate (map, line, nbr, z);
    if (test == 101)
        return (101);
    return (10);
}