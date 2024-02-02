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

int check(char **map, int z)
{
    int a = 0;
    for(int i = 0; i < (z + 1); ++i)
        for (int j = 0; map[i][j] != '\0'; ++j)
            if (map[i][j] == '|')
                ++a;
    return (a);
}